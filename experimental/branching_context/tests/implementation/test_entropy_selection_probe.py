"""CPU guards for the real-native entropy-selection audit receipt."""
import unittest

from run_entropy_selection_probe import policy_cases, selection_arguments, validate_receipt


NATIVE_PROTOCOL = {"v2": "full-vocab-min-p-u53-v2",
                  "v3": "full-vocab-dry-min-p-u53-v3",
                  "v4": "full-vocab-typical-dyna-min-p-u53-v4"}


def valid_receipt(include_dry=True):
    cases = policy_cases(include_dry=include_dry)
    rows = []
    v2 = v3 = v4 = 0
    for case in cases:
        protocol = case["expected_protocol"]
        for draw in case["draws"]:
            before = dict(sampling_calls=v2 + v3 + v4,
                          sampling_v2_calls=v2, sampling_v3_calls=v3,
                          sampling_v4_calls=v4, forward_tokens=11,
                          prefill_tokens=7743)
            if protocol == "v2": v2 += 1
            elif protocol == "v3": v3 += 1
            else: v4 += 1
            after = dict(sampling_calls=v2 + v3 + v4,
                         sampling_v2_calls=v2, sampling_v3_calls=v3,
                         sampling_v4_calls=v4, forward_tokens=11,
                         prefill_tokens=7743)
            rows.append(dict(name=case["name"], mask_kind=case["mask_kind"], policy=case["policy"],
                             protocol=protocol, draw53=draw, token=4,
                             reference_token=4, counters_before=before,
                             counters_after=after, effective_temperature=0.8,
                             typical_kept_count=3, legal_count=4,
                             native_response={"status": "ok", "token": 4,
                                              "protocol": NATIVE_PROTOCOL[protocol],
                                              "effective_temperature": 0.8,
                                              "typical_kept_count": 3},
                             dry_penalized_candidates=1 if case["name"] == "dry-combined" else 0))
    return dict(status="PASS", include_dry=include_dry, cases=cases, rows=rows,
                frontier_before_sha256="a" * 64, frontier_after_sha256="a" * 64,
                parent_before_sha256="b" * 64, parent_after_sha256="b" * 64,
                forward_tokens_before=11, forward_tokens_after=11,
                prefill_tokens_before=7743, prefill_tokens_after=7743,
                selection_forward_tokens=0,
                selection_counters_before=dict(sampling_calls=0, sampling_v2_calls=0,
                    sampling_v3_calls=0, sampling_v4_calls=0, forward_tokens=11, prefill_tokens=7743),
                selection_counters_after=dict(sampling_calls=v2 + v3 + v4, sampling_v2_calls=v2,
                    sampling_v3_calls=v3, sampling_v4_calls=v4, forward_tokens=11, prefill_tokens=7743))


class EntropySelectionProbeTests(unittest.TestCase):
    def test_trace_is_configured_after_the_real_profile_environment_helper(self):
        import os
        from unittest.mock import patch

        import grammar_long_reference
        import run_entropy_selection_probe as probe

        trace_path = "/tmp/selection-probe-test/native-trace.jsonl"
        profile = {"execution": {"environment": {
            "STRATA_SELECTION_TEST": "from-profile",
            "CTT_NATIVE_TRACE": "profile-value-must-not-win",
        }}}
        with patch.dict(os.environ, {
            "CTT_NATIVE_TRACE": "stale-value",
            "STRATA_SELECTION_TEST": "old-value",
        }, clear=False):
            probe.configure_native_trace(profile, trace_path,
                                         grammar_long_reference._apply_profile_environment)
            self.assertEqual(os.environ.get("STRATA_SELECTION_TEST"), "from-profile")
            self.assertEqual(os.environ.get("CTT_NATIVE_TRACE"), trace_path)

    def test_dry_entropy_arguments_are_assembled_once_for_both_call_paths(self):
        from context_time_travel import min_p

        policy = next(case["policy"] for case in policy_cases(include_dry=True)
                      if case["name"] == "dry-combined")
        dry_context = {
            "dry_multiplier": policy["dry_multiplier"],
            "dry_base": policy["dry_base"],
            "dry_allowed_length": policy["dry_allowed_length"],
            "dry_penalty_last_n": policy["dry_penalty_last_n"],
            "dry_history": (2, 3, 2, 4),
            "dry_breakers": (4,),
        }
        args = selection_arguments(policy, dry_context)
        self.assertEqual(args["dry_history"], dry_context["dry_history"])
        self.assertEqual(args["dry_breakers"], dry_context["dry_breakers"])
        self.assertEqual(args["typical_p"], 0.95)
        self.assertEqual(args["dynatemp_range"], 0.35)
        self.assertEqual(args["dry_multiplier"], 0.8)

        logits = [float(8 - token) for token in range(8)]
        mask = bytes([0xff])
        reference = min_p.choose(logits, mask, draw53=1 << 52, **args)

        class NativeCallCapture:
            def sample_legal(self, lease, native_mask, temperature, min_p_value,
                             draw53, **kwargs):
                return dict(lease=lease, mask=native_mask, temperature=temperature,
                            min_p=min_p_value, draw53=draw53, kwargs=kwargs)

        native_args = {key: value for key, value in args.items()
                       if key not in ("temperature", "min_p")}
        captured = NativeCallCapture().sample_legal(
            "lease", mask, args["temperature"], args["min_p"], 1 << 52,
            **native_args)
        self.assertEqual(captured["temperature"], args["temperature"])
        self.assertEqual(captured["min_p"], args["min_p"])
        self.assertEqual(captured["kwargs"], {
            key: value for key, value in args.items()
            if key not in ("temperature", "min_p")
        })
        self.assertTrue(0 <= reference["token"] < len(logits))

    def test_dry_context_numeric_mismatch_is_rejected(self):
        policy = next(case["policy"] for case in policy_cases(include_dry=True)
                      if case["name"] == "dry-combined")
        context = {key: policy[key] for key in (
            "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n")}
        context["dry_multiplier"] += 0.1
        context["dry_history"] = (1, 2)
        context["dry_breakers"] = ()
        with self.assertRaisesRegex(ValueError, "disagrees with canonical dry_multiplier"):
            selection_arguments(policy, context)

    def test_policy_inventory_covers_disabled_and_individual_entropy_presets(self):
        cases = policy_cases(include_dry=True)
        self.assertEqual(
            [case["name"] for case in cases],
            ["greedy", "min-p-baseline", "typical-alone", "dynatemp-alone",
             "typical-dynatemp", "dry-combined", "threshold-corner", "single-legal"],
        )
        self.assertEqual(cases[0]["draws"], [0])
        for case in cases[1:7]:
            self.assertEqual(case["draws"], [0, 1 << 52, (1 << 53) - 1])
        self.assertEqual(cases[7]["draws"], [0, 1 << 52, (1 << 53) - 1])
        self.assertEqual(cases[2]["policy"]["temperature"], 0.8)
        self.assertEqual(cases[2]["policy"]["min_p"], 0.0)
        self.assertEqual(cases[2]["policy"]["typical_p"], 0.95)
        self.assertEqual(cases[3]["policy"]["temperature"], 0.55)
        self.assertEqual(cases[3]["policy"]["dynatemp_range"], 0.35)
        self.assertEqual(cases[3]["policy"]["dynatemp_exponent"], 1.0)
        self.assertEqual(cases[3]["policy"]["min_p"], 0.06)
        self.assertEqual(cases[4]["policy"]["typical_p"], 0.95)
        self.assertEqual(cases[4]["policy"]["dynatemp_range"], 0.35)

    def test_receipt_accepts_one_native_selection_counter_per_row(self):
        self.assertTrue(validate_receipt(valid_receipt()))

    def test_receipt_rejects_frontier_change_or_forward_work(self):
        import copy
        receipt = valid_receipt()
        receipt["frontier_after_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_receipt(receipt)
        receipt["frontier_after_sha256"] = receipt["frontier_before_sha256"]
        receipt["forward_tokens_after"] = 12
        with self.assertRaises(ValueError):
            validate_receipt(receipt)
        receipt = copy.deepcopy(valid_receipt())
        receipt["parent_after_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_receipt(receipt)

    def test_receipt_rejects_v4_double_counting_in_v2_or_v3(self):
        import copy
        receipt = copy.deepcopy(valid_receipt())
        row = next(row for row in receipt["rows"] if row["protocol"] == "v4")
        row["counters_after"]["sampling_v2_calls"] += 1
        row["counters_after"]["sampling_calls"] += 1
        with self.assertRaises(ValueError):
            validate_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
