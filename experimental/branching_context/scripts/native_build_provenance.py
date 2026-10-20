"""Additive, fail-closed and explicitly incomplete provenance for native probes.

This records observed build inputs without certifying hermetic reproduction or
numerical behavior. External toolchain and host closure remains unproven.
"""
import ast
import hashlib
from pathlib import Path
import re
import shlex
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root):
    try:
        revision = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {'status': 'non_git', 'revision': None, 'tracked_changes': [], 'untracked': []}
    status = subprocess.run(
        ['git', '-C', str(root), 'status', '--porcelain=v1', '--untracked-files=all'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    tracked, untracked = [], []
    for line in status:
        name = line[3:] if len(line) > 3 else line
        if line.startswith('??'):
            untracked.append(name)
        else:
            tracked.append(name)
    return {'status': 'clean' if not status else 'dirty', 'revision': revision,
            'tracked_changes': sorted(tracked), 'untracked': sorted(untracked)}


def _local_python_import_closure(entrypoints):
    pending = [Path(path).resolve() for path in entrypoints]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f'missing generator source: {path}')
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (UnicodeError, SyntaxError) as error:
            raise ValueError(f'cannot inspect generator source {path}: {error}') from error
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            for name in names:
                candidate = path.parent.joinpath(*name.split('.')).with_suffix('.py')
                package = path.parent.joinpath(*name.split('.')) / '__init__.py'
                if candidate.is_file():
                    pending.append(candidate)
                elif package.is_file():
                    pending.append(package)
    return {str(path): sha256_file(path) for path in sorted(seen)}


def _response_expanded(argv, cwd):
    expanded = []
    response_hashes = {}
    pending = list(argv)
    while pending:
        arg = pending.pop(0)
        if arg.startswith('@') and len(arg) > 1:
            response = Path(arg[1:])
            if not response.is_absolute():
                response = cwd / response
            response = response.resolve()
            if response in response_hashes:
                raise ValueError(f'recursive/repeated link response file: {response}')
            if not response.is_file():
                raise FileNotFoundError(f'missing link response file: {response}')
            response_hashes[response] = sha256_file(response)
            nested = shlex.split(response.read_text(encoding='utf-8'), posix=True)
            pending = nested + pending
        else:
            expanded.append(arg)
    return expanded, response_hashes


def _link_inputs(commands, link_working_directory):
    cwd = Path(link_working_directory).resolve()
    archives, responses = {}, {}
    link_count = 0
    for index, record in enumerate(commands):
        if not isinstance(record, dict) or not isinstance(record.get('argv'), list):
            raise ValueError(f'build command {index} must be a receipt object with argv and exit_code')
        if record.get('exit_code') != 0:
            raise ValueError(f'build command {index} did not succeed')
        command = record['argv']
        if not command or any(not isinstance(arg, str) for arg in command):
            raise ValueError(f'build command {index} has malformed argv')
        # Generated probe commands identify final links by their executable
        # output names. Do not mistake compiler -c commands for link recipes.
        if '-c' in command:
            continue
        if '-o' not in command:
            continue
        link_count += 1
        args, response_files = _response_expanded(command, cwd)
        for path, digest in response_files.items():
            responses[str(path)] = digest
        for arg in args:
            if arg.endswith(('.a', '.so')) or re.search(r'\.so\.\d+(?:\.\d+)*$', arg):
                path = Path(arg)
                if not path.is_absolute():
                    path = cwd / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f'missing linked input: {path}')
                archives[str(path)] = sha256_file(path)
    if not link_count:
        raise ValueError('no final link command found in build receipt')
    return archives, responses


def _cmake_toolchain(build_dir):
    cache = Path(build_dir) / 'CMakeCache.txt'
    if not cache.is_file():
        raise FileNotFoundError(f'missing CMake cache: {cache}')
    values = {}
    for line in cache.read_text(encoding='utf-8', errors='strict').splitlines():
        if ':' not in line or '=' not in line or line.startswith('//') or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        name, _, kind = key.partition(':')
        if (name.startswith(('CMAKE_', 'STRATA_')) and
                any(part in name for part in ('COMPILER', 'BUILD_TYPE', 'CUDA_ARCH', 'GENERATOR',
                                               'TOOLCHAIN', 'CUDA', 'GGML', 'NATIVE', 'PREFILL',
                                               'TESTS', 'FLAGS', 'LINKER'))):
            values[name] = {'type': kind, 'value': value}
    return {'cache_path': str(cache.resolve()), 'cache_sha256': sha256_file(cache),
            'selected_cache_values': dict(sorted(values.items()))}


def collect_build_provenance(*, source_root, build_dir, commands,
                             generator_entrypoints, declared_sources=(),
                             link_working_directory=None):
    root = Path(source_root).resolve()
    build = Path(build_dir).resolve()
    git = _git_state(root)
    generator_files = _local_python_import_closure(generator_entrypoints)

    sources = {}
    for item in declared_sources:
        path = Path(item).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'missing declared build input: {path}')
        sources[str(path)] = sha256_file(path)

    # Bind CMake's effective compile and link recipes used by these probe
    # builders, in addition to the cache itself.
    for relative in ('CMakeFiles/strata-branch.dir/flags.make',
                     'CMakeFiles/strata-branch.dir/link.txt'):
        path = build / relative
        if not path.is_file():
            raise FileNotFoundError(f'missing CMake build recipe: {path}')
        sources[str(path)] = sha256_file(path)

    archives, response_files = _link_inputs(
        commands, link_working_directory or build)
    toolchain = _cmake_toolchain(build)
    # This receipt pins every input observed by these helpers. It deliberately
    # does not claim hermetic reproducibility: system headers, compiler binaries,
    # and libraries named only by linker search name are not closed here.
    observed_inputs_complete = bool(generator_files and sources and archives and response_files is not None)
    return {
        'schema': 'strata-native-build-provenance-v1',
        'source': git,
        'cmake_toolchain': toolchain,
        'generator_closure': {'method': 'static-local-python-import-closure',
                              'files': generator_files},
        'declared_build_inputs': dict(sorted(sources.items())),
        'linked_archives': dict(sorted(archives.items())),
        'link_response_files': dict(sorted(response_files.items())),
        'observed_inputs_complete': observed_inputs_complete,
        'provenance_complete': False,
        'reproducible': False,
        'reproducibility_status': 'unproven_external_toolchain_closure',
        'limitations': [
            'Only local Python imports reachable by static import syntax are included.',
            'External compiler binaries, system headers, dynamic libraries named only with -l, and host state are not hashed.',
            'A dirty or non-Git source root is historical/non-reproducible even when all observed file hashes are recorded.',
        ],
    }
