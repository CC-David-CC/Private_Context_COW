"""Build an isolated ordinary ``strata --serve`` binary from pinned COW objects.

This builder never edits the source checkout or an existing CMake build tree.
The output is an unqualified native baseline until separately exercised on GPU.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StandardRecipe:
    target: str
    build_dir: Path
    cache_path: Path
    flags_path: Path
    link_path: Path
    flags: dict
    link_argv: list
    generate_object: str
    archives: tuple


def _cmake_flags(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
            key, value = line.split("=", 1)
            result[key.strip()] = shlex.split(value)
    missing = {"CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"} - set(result)
    if missing:
        raise ValueError(f"incomplete standard compile flags: {sorted(missing)}")
    return result


def load_standard_recipe(build_dir):
    """Load and fail closed unless this is the ordinary CMake strata target."""
    build_dir = Path(build_dir).resolve()
    cache = build_dir / "CMakeCache.txt"
    target_dir = build_dir / "CMakeFiles/strata.dir"
    flags_path = target_dir / "flags.make"
    link_path = target_dir / "link.txt"
    for path in (cache, flags_path, link_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing ordinary CMake build input: {path}")
    flags = _cmake_flags(flags_path)
    if any("STRATA_BRANCH_DRIVER" in item
           for group in flags.values() for item in group):
        raise ValueError("ordinary strata target must not define STRATA_BRANCH_DRIVER")
    argv = shlex.split(link_path.read_text())
    if not argv or argv.count("-o") != 1:
        raise ValueError(f"invalid ordinary strata link recipe: {link_path}")
    out_index = argv.index("-o")
    if out_index + 1 >= len(argv) or argv[out_index + 1] != "strata":
        raise ValueError("expected ordinary strata target output")
    generate_object = "CMakeFiles/strata.dir/src/program/generate.cpp.o"
    if argv.count(generate_object) != 1:
        raise ValueError("ordinary strata recipe must contain standard generate.cpp exactly once")
    if any("branch_driver" in token for token in argv):
        raise ValueError("ordinary strata recipe must not link branch_driver")
    if any("strata-branch" in token for token in argv):
        raise ValueError("expected ordinary strata target, not strata-branch")
    archives = tuple(token for token in argv if token.endswith((".a", ".so")))
    if not archives:
        raise ValueError("ordinary link recipe has no static/shared library closure")
    for token in argv[1:out_index]:
        if token.endswith(".o"):
            path = Path(token)
            if not path.is_absolute():
                path = build_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"missing ordinary target object: {path}")
    for token in archives:
        path = Path(token)
        if not path.is_absolute():
            path = build_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"missing ordinary link archive/library: {path}")
    return StandardRecipe("strata", build_dir, cache, flags_path, link_path,
                          flags, argv, generate_object, archives)


CORE_OBJECTS = (
    ("native-probe-v3-core/branch_digest.o", "src/core/branch_digest.cu"),
    ("native-probe-v3-core/branch_pages.o", "src/core/branch_pages.cpp"),
    ("native-probe-v3-core/vmm.o", "src/core/vmm.cpp"),
    ("native-probe-v3-core/layer.o", "src/core/layer.cpp"),
    ("native-probe-v3-core/expert_cache.o", "src/core/expert_cache.cpp"),
    ("native-probe-v3/branch_session.o", "src/core/branch_session.cpp"),
    ("native-probe-v3/conversation_snapshot.o", "src/core/conversation_snapshot.cpp"),
    ("native-probe-v3/conversation_state.o", "src/core/conversation_state.cpp"),
    ("native-probe-v3/conversation_file.o", "src/core/conversation_file.cpp"),
)


def current_override_objects(cow_build_root):
    """Return the reviewed current COW objects; never include private driver.o."""
    root = Path(cow_build_root).resolve()
    result = []
    for relative, _source in CORE_OBJECTS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing current COW override object: {path}")
        result.append(path)
    return result


def verify_cow_build_receipt(cow_build_root):
    root = Path(cow_build_root).resolve()
    receipt_path = root / "native-probe-v3/build.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing reviewed current COW object receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "BUILT_NOT_EXECUTED":
        raise ValueError("current COW object receipt must be BUILT_NOT_EXECUTED")
    verified = {str(receipt_path): sha256(receipt_path)}
    for field in ("source_sha256", "generated_sha256"):
        entries = receipt.get(field)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"COW object receipt lacks {field}")
        for raw_path, expected in entries.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(f"COW receipt input is missing: {path}")
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"COW receipt hash mismatch for {path}")
            verified[str(path)] = actual
    commands = receipt.get("commands")
    if not isinstance(commands, list):
        raise ValueError("COW object receipt lacks command rows")
    for object_relative, source_relative in CORE_OBJECTS:
        source = (root / "source" / source_relative).resolve()
        obj = (root / object_relative).resolve()
        matches = []
        for row in commands:
            if not isinstance(row, dict) or row.get("exit_code") != 0:
                continue
            argv = row.get("argv")
            if not isinstance(argv, list) or "-o" not in argv:
                continue
            out_index = argv.index("-o")
            if out_index + 1 < len(argv) and Path(argv[out_index + 1]).resolve() == obj and str(source) in argv:
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(f"COW receipt must pin one successful compile for {source} -> {obj}")
        verified[str(obj)] = sha256(obj)
    return receipt_path, verified


def make_commands(recipe, source_root, generate_source, generated_object, output, overrides):
    source_root = Path(source_root).resolve()
    generate_source = Path(generate_source).resolve()
    generated_object = Path(generated_object).resolve()
    output = Path(output).resolve()
    if not generate_source.is_file():
        raise FileNotFoundError(f"missing standard generate.cpp: {generate_source}")
    if len(overrides) != len(CORE_OBJECTS):
        raise ValueError(f"expected exactly {len(CORE_OBJECTS)} current COW override objects")
    if any(not Path(path).is_file() for path in overrides):
        raise FileNotFoundError("missing current COW override object")
    compiler = recipe.link_argv[0]
    includes = list(recipe.flags["CXX_INCLUDES"])
    project_include_positions = [i for i, token in enumerate(includes)
                                 if token.startswith("-I") and
                                 Path(token[2:]).name == "include" and
                                 not any(part in token for part in ("llama-pinned", "ggml"))]
    if len(project_include_positions) != 1:
        raise ValueError("expected exactly one ordinary project include directory")
    includes[project_include_positions[0]] = "-I" + str(source_root / "include")
    compile_argv = [compiler, *recipe.flags["CXX_DEFINES"],
                    *includes, *recipe.flags["CXX_FLAGS"],
                    "-c", str(generate_source), "-o", str(generated_object)]
    if any("STRATA_BRANCH_DRIVER" in arg for arg in compile_argv):
        raise ValueError("standard generate compile unexpectedly enables branch driver")
    link_argv = list(recipe.link_argv)
    index = link_argv.index(recipe.generate_object)
    link_argv[index] = str(generated_object)
    insertions = []
    for (object_relative, source_relative), override in zip(CORE_OBJECTS, overrides):
        standard_object = f"CMakeFiles/strata.dir/{source_relative}.o"
        matches = [i for i, token in enumerate(link_argv)
                   if token == standard_object or token.endswith("/" + source_relative + ".o")]
        if len(matches) > 1:
            raise ValueError(f"duplicate ordinary object for COW override source: {source_relative}")
        if matches:
            link_argv[matches[0]] = str(Path(override).resolve())
        else:
            insertions.append(str(Path(override).resolve()))
    link_argv[index + 1:index + 1] = insertions
    out_index = link_argv.index("-o")
    link_argv[out_index + 1] = str(output)
    if any("branch_driver" in arg or "strata-branch" in arg for arg in link_argv):
        raise ValueError("private branch target/driver leaked into ordinary baseline")
    return compile_argv, link_argv


def prepare_output(output):
    output = validate_output_path(output)
    output.mkdir(parents=True)
    return output


def validate_output_path(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    return output


def plan_build(source_root, standard_build_dir, cow_build_root, output):
    """Resolve and validate every compile/link input without creating output."""
    source_root = Path(source_root).resolve()
    candidate_root = Path(cow_build_root).resolve()
    if source_root != candidate_root / "source":
        raise ValueError("source root must be the pinned candidate's source directory")
    recipe = load_standard_recipe(standard_build_dir)
    generate_source = source_root / "src/program/generate.cpp"
    overrides = current_override_objects(cow_build_root)
    output = validate_output_path(output)
    if not generate_source.is_file():
        raise FileNotFoundError(f"missing standard generate.cpp: {generate_source}")
    candidate_receipt, cow_verified = verify_cow_build_receipt(candidate_root)
    sources = [source_root / source for _obj, source in CORE_OBJECTS]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"missing current COW object source: {path}")
    generated_object = output / "generate.o"
    executable = output / "strata-normal-baseline"
    compile_argv, link_argv = make_commands(
        recipe, source_root, generate_source, generated_object, executable, overrides)
    inputs = [recipe.cache_path, recipe.flags_path, recipe.link_path,
              generate_source, *overrides, *sources]
    for archive in recipe.archives:
        path = Path(archive)
        inputs.append(path if path.is_absolute() else recipe.build_dir / path)
    return {
        "builder_sha256": sha256(Path(__file__)),
        "source_root": str(source_root),
        "standard_build_dir": str(recipe.build_dir),
        "cow_object_root": str(Path(cow_build_root).resolve()),
        "output": str(output),
        "compile_argv": compile_argv,
        "link_argv": link_argv,
        "override_objects": [str(path) for path in overrides],
        "inputs_sha256": {**cow_verified, **{str(path): sha256(path) for path in inputs}},
        "recipe": recipe,
    }


def build(source_root, standard_build_dir, cow_build_root, output):
    plan = plan_build(source_root, standard_build_dir, cow_build_root, output)
    output = prepare_output(plan["output"])
    recipe = plan["recipe"]
    compile_argv = plan["compile_argv"]
    link_argv = plan["link_argv"]
    overrides = plan["override_objects"]
    source_root = Path(plan["source_root"])
    inputs_sha256 = plan["inputs_sha256"]
    executable = output / "strata-normal-baseline"
    receipt = {
        "schema": "normal-serving-baseline-build-v1",
        "status": "BUILDING",
        "qualification": "UNQUALIFIED_UNTIL_GPU_SMOKE",
        "serving_scope": "ordinary-serving-on-current-patched-engine",
        "builder_sha256": plan["builder_sha256"],
        "source_root": plan["source_root"],
        "standard_build_dir": plan["standard_build_dir"],
        "cow_object_root": plan["cow_object_root"],
        "output": str(output),
        "inputs_sha256": inputs_sha256,
        "override_objects": overrides,
        "compile_argv": compile_argv,
        "link_argv": link_argv,
        "commands": [],
    }
    receipt_path = output / "build.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for label, argv, cwd in (("compile", compile_argv, source_root),
                             ("link", link_argv, recipe.build_dir)):
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        receipt["commands"].append({"name": label, "argv": argv, "cwd": str(cwd),
                                    "exit_code": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})
        receipt["status"] = "BUILD_FAILED" if result.returncode else "BUILDING"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv,
                                                result.stdout, result.stderr)
    receipt["executable_sha256"] = sha256(executable)
    receipt["status"] = "BUILT_UNQUALIFIED"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("standard_build_dir", type=Path)
    parser.add_argument("cow_build_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-only", action="store_true",
                        help="print exact pinned inputs and argv without creating output or compiling")
    args = parser.parse_args(argv)
    if args.plan_only:
        plan = plan_build(args.source_root, args.standard_build_dir,
                          args.cow_build_root, args.output)
        plan.pop("recipe")
        print(json.dumps(plan, indent=2))
        return 0
    build(args.source_root, args.standard_build_dir, args.cow_build_root, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
