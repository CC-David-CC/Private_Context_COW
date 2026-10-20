"""Write execution profiles only after all fingerprinted metadata is final."""
import hashlib
import json
import copy
from pathlib import Path


def execution_profile_id(profile):
    value = {
        "build": profile["native_build"],
        "execution": profile["execution"],
        "hardware": profile["hardware"],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finalize_execution_profile(profile, path, *, validator=None, mode=None):
    """Seal, serialize, reload, and validate the exact profile that will launch.

    Callers must finish edits to native_build, execution, and hardware before
    calling. Validation runs on the reloaded JSON value, so post-digest changes
    to fingerprinted metadata cannot pass accidentally.
    """
    path = Path(path)
    if "execution_profile_id" not in profile:
        raise ValueError("profile is missing execution_profile_id field")
    profile["execution_profile_id"] = execution_profile_id(profile)
    data = json.dumps(profile, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    created = False
    try:
        with path.open("x", encoding="ascii", newline="\n") as stream:
            created = True
            stream.write(data)
        serialized = json.loads(path.read_text(encoding="ascii"))
        if serialized["execution_profile_id"] != execution_profile_id(serialized):
            raise ValueError("serialized execution profile fingerprint mismatch")
        if validator is not None:
            validator(copy.deepcopy(serialized))
        if mode is not None:
            path.chmod(mode)
        # Re-read and validate the exact final path after permissions are set.
        serialized = json.loads(path.read_text(encoding="ascii"))
        if serialized["execution_profile_id"] != execution_profile_id(serialized):
            raise ValueError("final execution profile fingerprint mismatch")
        if validator is not None:
            validator(copy.deepcopy(serialized))
        return serialized
    except BaseException:
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
