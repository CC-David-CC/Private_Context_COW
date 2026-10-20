"""Stage a new driver without changing the existing source, objects or binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or base in output.parents:
        raise ValueError('Stage outside the frozen source tree')
    output.mkdir(parents=True, exist_ok=False)
    (output/'include').symlink_to(base/'include', target_is_directory=True)
    (output/'build').symlink_to(base/'build', target_is_directory=True)
    (output/'src').mkdir()
    (output/'src/program').symlink_to(base/'src/program', target_is_directory=True)
    core = output/'src/core'
    core.mkdir()
    for path in (base/'src/core').iterdir():
        if path.is_file() and path.name != 'branch_driver.cpp':
            (core/path.name).symlink_to(path)
    shutil.copyfile(args.driver, core/'branch_driver.cpp')
    record = dict(base_source=str(base), staged_source=str(output),
        driver_sha256=hashlib.sha256((core/'branch_driver.cpp').read_bytes()).hexdigest(),
        policy='reuse frozen objects; new source and probe outputs only')
    (output/'stage.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
