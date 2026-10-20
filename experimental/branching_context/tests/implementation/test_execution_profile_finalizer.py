import json
import stat
import tempfile
import unittest
from pathlib import Path

from execution_profile_finalizer import execution_profile_id, finalize_execution_profile


class ExecutionProfileFinalizerTests(unittest.TestCase):
    def _profile(self):
        return {
            "execution_profile_id": "",
            "native_build": {"source_status": "final-status", "sha256": {"native": "abc"}},
            "execution": {"environment": {}, "capacity": 4096},
            "hardware": "pinned-device",
        }

    def _validate(self, profile):
        if profile["execution_profile_id"] != execution_profile_id(profile):
            raise ValueError("execution profile fingerprint mismatch")

    def test_finalizer_validates_the_serialized_final_metadata(self):
        profile = self._profile()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            observed = []
            result = finalize_execution_profile(profile, path, validator=observed.append)
            self.assertEqual(result, json.loads(path.read_text(encoding="ascii")))
            self.assertEqual(observed, [result, result])
            self.assertEqual(result["native_build"]["source_status"], "final-status")
            self._validate(result)

    def test_post_digest_fingerprinted_mutation_is_rejected(self):
        profile = self._profile()
        profile["native_build"]["source_status"] = "pre-digest-status"
        profile["execution_profile_id"] = execution_profile_id(profile)
        expected_id = profile["execution_profile_id"]
        profile["native_build"]["source_status"] = "post-digest-status"
        self.assertNotEqual(execution_profile_id(profile), expected_id)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.json"
            path.write_text(json.dumps(profile), encoding="ascii")
            serialized = json.loads(path.read_text(encoding="ascii"))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                self._validate(serialized)

    def test_finalizer_removes_profile_if_serialized_validator_rejects(self):
        profile = self._profile()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rejected.json"

            def reject(_profile):
                raise ValueError("intentional validation refusal")

            with self.assertRaisesRegex(ValueError, "intentional validation refusal"):
                finalize_execution_profile(profile, path, validator=reject)
            self.assertFalse(path.exists())

    def test_validator_cannot_mutate_the_written_profile_or_return_value(self):
        profile = self._profile()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"

            def mutating_validator(value):
                value["native_build"]["source_status"] = "validator-local mutation"

            result = finalize_execution_profile(profile, path, validator=mutating_validator)
            on_disk = json.loads(path.read_text(encoding="ascii"))
            self.assertEqual(result, on_disk)
            self.assertEqual(result["native_build"]["source_status"], "final-status")
            self.assertEqual(result["execution_profile_id"], execution_profile_id(result))

    def test_corrected_finalization_binds_the_final_status(self):
        profile = self._profile()
        profile["native_build"]["source_status"] = "post-digest-status"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "correct.json"
            result = finalize_execution_profile(profile, path, validator=self._validate)
            self.assertEqual(result["native_build"]["source_status"], "post-digest-status")
            self.assertEqual(result["execution_profile_id"], execution_profile_id(result))

    def test_destination_must_be_new(self):
        profile = self._profile()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            original = b"existing profile must be preserved\n"
            path.write_bytes(original)
            path.chmod(0o440)
            before_mode = stat.S_IMODE(path.stat().st_mode)
            with self.assertRaises(FileExistsError):
                finalize_execution_profile(profile, path, validator=self._validate)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), before_mode)


if __name__ == "__main__":
    unittest.main()
