"""CPU guards that resumed CLI commands preserve entropy sampler settings."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_sampler_serving_probe import sampling_arguments
from context_time_travel.min_p import validate_policy


CORE = {
    "temperature": 0.8,
    "min_p": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


class TypicalDynaTempObserverTests(unittest.TestCase):
    def test_resume_emits_changed_typical_and_dynamic_temperature_fields_only(self):
        previous = validate_policy(dict(CORE, typical_p=0.99,
                                        dynatemp_range=0.1,
                                        dynatemp_exponent=1.2))
        resumed = validate_policy(dict(CORE, typical_p=0.95,
                                       dynatemp_range=0.25,
                                       dynatemp_exponent=1.5))
        self.assertEqual(
            sampling_arguments(resumed, previous),
            ["--typical-p", "0.95",
             "--dynatemp-range", "0.25",
             "--dynatemp-exponent", "1.5"],
        )

    def test_unchanged_entropy_settings_are_inherited_without_cli_flags(self):
        policy = validate_policy(dict(CORE, typical_p=0.95,
                                      dynatemp_range=0.25,
                                      dynatemp_exponent=1.5))
        self.assertEqual(sampling_arguments(policy, policy), [])

    def test_starting_optional_entropy_controls_emits_normalized_values(self):
        previous = validate_policy(dict(CORE))
        # validate_policy fills DynaTemp's exponent default when its range is
        # first enabled; the CLI resume must receive both values explicitly.
        resumed = validate_policy(dict(CORE, typical_p=0.9,
                                       dynatemp_range=0.2))
        self.assertEqual(
            sampling_arguments(resumed, previous),
            ["--typical-p", "0.9",
             "--dynatemp-range", "0.2",
             "--dynatemp-exponent", "1.0"],
        )


if __name__ == "__main__":
    unittest.main()
