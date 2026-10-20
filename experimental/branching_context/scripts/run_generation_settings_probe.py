"""Bounded same-schedule audit of per-generation sampling settings.

This tool creates fresh raw-native references for each policy, prepares one
canonical Q4/Q8 question snapshot, then evaluates all candidate policies from
that hot snapshot in one native process. It is a correctness harness, not a
throughput qualification or user-facing CLI benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


DEFAULT_POLICIES = [
    {"name": "baseline", "sampling": {"temperature": 0.7, "min_p": 0.05, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "temperature-zero", "sampling": {"temperature": 0.0, "min_p": 0.05, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "temperature-low", "sampling": {"temperature": 0.4, "min_p": 0.05, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "temperature-high", "sampling": {"temperature": 1.0, "min_p": 0.05, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "min-p-off", "sampling": {"temperature": 0.7, "min_p": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "top-k-40", "sampling": {"temperature": 0.7, "min_p": 0.05, "top_k": 40, "top_p": 1.0, "repetition_penalty": 1.0}},
    {"name": "top-p-09", "sampling": {"temperature": 0.7, "min_p": 0.05, "top_k": 0, "top_p": 0.9, "repetition_penalty": 1.0}},
]
STREAM = "generation-settings-audit"


def reference_selector(logits, policy):
    """Host reference uses its recorded policy, with the session call signature."""
    from context_time_travel import min_p
    def select(mask, temperature, probability, draw, **filters):
        expected={} if policy['top_k']==0 and policy['top_p']==1.0 else {
            'top_k':policy['top_k'],'top_p':policy['top_p']}
        if filters != expected:
            raise AssertionError('Reference session supplied filters inconsistent with its recorded policy')
        return min_p.choose(logits,mask,temperature=temperature,min_p=probability,draw53=draw,
                            top_k=policy['top_k'],top_p=policy['top_p'])['token']
    return select


def continuation_samples(samples):
    """Select reachable boundaries; preparation separately checks the prefix."""
    if 'prefix' not in samples or 'question' not in samples:
        raise AssertionError('Fresh reference must include both prepared boundaries')
    return {key:value for key,value in samples.items() if key!='prefix'}


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _finite_frontier(raw, vocab):
    if len(raw) != vocab * 4:
        raise AssertionError("Native frontier has the wrong full-vocabulary extent")
    values = struct.unpack("<" + "f" * vocab, raw)
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("Native frontier contains nonfinite logits")


def _frontier(raw, directory, name, vocab):
    _finite_frontier(raw, vocab)
    path = Path(directory) / (name + ".f32")
    path.write_bytes(raw)
    return {"path": path.name, "bytes": len(raw), "sha256": _sha(path)}


def _equal(left, right, label):
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if canonical(left) != canonical(right):
        raise AssertionError("Exact same-schedule mismatch: " + label)


def _check_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"temperature", "min_p", "top_k", "top_p", "repetition_penalty"}:
        raise ValueError("Sampling policy must contain exactly five supported fields")
    for key in ("temperature", "min_p", "top_p", "repetition_penalty"):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("Sampling policy contains a nonfinite or nonnumeric field")
    if (policy["temperature"] < 0 or not 0 <= policy["min_p"] <= 1
            or type(policy["top_k"]) is not int or not 0 <= policy["top_k"] <= (1 << 64) - 1
            or not 0 < policy["top_p"] <= 1 or policy["repetition_penalty"] != 1):
        raise ValueError("Sampling policy is outside the supported ranges")


def _record_selection(session, logits, mask, selector, tokenizer, raw, directory, vocab, position, samples, tokens):
    token = session.choose(mask, selector)
    receipt = raw.request("decode " + str(token), "decode")
    if receipt.get("position") != position:
        raise AssertionError("Raw reference decode receipt position differs")
    session._consume_selected(token, tokenizer.token_bytes(token), token in session._state["policy"]["eos_tokens"])
    position += 1
    tokens.append(token)
    samples["generated-" + str(len(tokens))] = _frontier(
        raw.request("logits", "logits", payload=directory / ("generated-" + str(len(tokens)) + ".f32")),
        directory, "generated-" + str(len(tokens)), vocab)
    return position


class CampaignObserver:
    """Monotonic, fail-closed progress aggregator used by CPU tests too."""

    def __init__(self, expected):
        if type(expected) is not int or expected <= 0:
            raise ValueError("Expected policy count must be positive")
        self.expected = expected
        self.rows = {}

    def update(self, name, *, phase, tokens, expected_tokens):
        if not isinstance(name, str) or not name or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in name):
            raise ValueError("Unsafe policy label")
        if phase not in ("reference", "candidate", "complete"):
            raise ValueError("Unknown campaign phase")
        if type(tokens) is not int or type(expected_tokens) is not int or not 0 <= tokens <= expected_tokens:
            raise ValueError("Invalid progress token count")
        prior = self.rows.get(name)
        if prior and (tokens < prior["tokens"] or expected_tokens != prior["expected_tokens"]):
            raise ValueError("Campaign progress cannot move backwards")
        self.rows[name] = {"phase": phase, "tokens": tokens, "expected_tokens": expected_tokens}

    def qualify(self, records):
        if len(records) != self.expected:
            return False
        names = set()
        for item in records:
            if not isinstance(item, dict) or item.get("status") != "PASS":
                return False
            code = item.get("native_exit_code")
            if type(code) is not int or code != 0 or item.get("native_pid_absent") is not True:
                return False
            if item.get("exact_match") is not True or item.get("reference_tokens") != item.get("candidate_tokens"):
                return False
            reference_draw = item.get("reference_next_draw")
            candidate_draw = item.get("candidate_next_draw")
            if (type(reference_draw) is not int or type(candidate_draw) is not int
                    or reference_draw != candidate_draw):
                return False
            name = item.get("name")
            if not isinstance(name, str) or name in names:
                return False
            names.add(name)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("campaign", "reference", "prepare", "candidate-sweep", "switch-reference", "switch-prepare", "switch-resume"), required=True)
    for field in ("runtime-root", "profile", "fixture", "run-dir"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--prefix", type=int, default=1024)
    parser.add_argument("--suffix", type=int, default=16)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("--switch-after", type=int, default=4)
    parser.add_argument("--policy-name")
    parser.add_argument("--switch-policy-name")
    args = parser.parse_args()
    if not (0 < args.prefix <= 8192 and 0 < args.suffix <= 512 and 1 <= args.outputs <= 512):
        raise ValueError("Fixture/output bounds exceeded")
    if type(args.switch_after) is not int or not 0 < args.switch_after < args.outputs:
        raise ValueError("Policy switch must be inside output scope")
    policies = _json(args.policies) if args.policies else DEFAULT_POLICIES
    if not isinstance(policies, list) or len(policies) < 2:
        raise ValueError("At least two policies are required")
    names = set()
    for row in policies:
        if not isinstance(row, dict) or set(row) != {"name", "sampling"} or not isinstance(row["name"], str):
            raise ValueError("Malformed named policy")
        _check_policy(row["sampling"])
        if row["name"] in names or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in row["name"]):
            raise ValueError("Policy names must be unique safe labels")
        names.add(row["name"])
    runtime = args.runtime_root.resolve(strict=True)
    output = args.run_dir.absolute()
    profile = _json(args.profile)

    if args.stage == "campaign":
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        script = Path(__file__).resolve()
        inventory = {"status": "RUNNING", "scope": __doc__, "script_sha256": _sha(script),
                     "profile_sha256": _sha(args.profile), "fixture_sha256": _sha(args.fixture),
                     "policies": policies, "prefix": args.prefix, "suffix": args.suffix,
                     "outputs": args.outputs, "stream": STREAM, "stages": []}
        _write(output / "campaign.json", inventory)
        plan = [("reference", row["name"], row["name"], None) for row in policies]
        plan += [("prepare", "shared-question", policies[0]["name"], None)]
        plan += [("candidate-sweep", "hot-question", policies[0]["name"], None)]
        plan += [("switch-reference", "switch-case", policies[0]["name"], policies[2]["name"]),
                 ("switch-prepare", "switch-case", policies[0]["name"], policies[2]["name"]),
                 ("switch-resume", "switch-case", policies[0]["name"], policies[2]["name"])]
        for stage, name, policy_name, switch_policy_name in plan:
            leaf = output / ("prepare-shared" if stage == "prepare" else stage + "-" + name)
            command = [sys.executable, "-B", str(script), "--stage", stage,
                       "--runtime-root", str(runtime), "--profile", str(args.profile.resolve()),
                       "--fixture", str(args.fixture.resolve()), "--run-dir", str(leaf),
                       "--prefix", str(args.prefix), "--suffix", str(args.suffix),
                       "--outputs", str(args.outputs), "--switch-after", str(args.switch_after)]
            command += ["--policy-name", policy_name]
            if switch_policy_name:
                command += ["--switch-policy-name", switch_policy_name]
            if args.policies:
                command += ["--policies", str(args.policies.resolve())]
            sys.path[:0] = [str(runtime), str(runtime / "tools"),
                str(runtime / "experimental/branching_context/src"),
                str(runtime / "experimental/branching_context/scripts")]
            from run_grammar_first_probe import run_stage
            receipt = run_stage(command, output / (stage + "-" + name + ".stdout"),
                                output / (stage + "-" + name + ".stderr"), timeout=900)
            child_path = leaf / "result.json"
            child = _json(child_path) if child_path.exists() else {"status": "MISSING"}
            row = {"stage": stage, "name": name, "command": command, **receipt,
                   "result_sha256": _sha(child_path) if child_path.exists() else None,
                   "status": child.get("status")}
            inventory["stages"].append(row)
            _write(output / "campaign.json", inventory)
            if (receipt["exit_code"] != 0 or not receipt["pid_absent"] or receipt["timed_out"]
                    or receipt["cleanup_needed"] or receipt["live_group_pids"]
                    or child.get("status") != "PASS" or child.get("native_exit_code") != 0
                    or child.get("native_pid_absent") is not True):
                inventory["status"] = "FAIL"
                _write(output / "campaign.json", inventory)
                raise AssertionError("Campaign stage failed: " + stage + "/" + name)
        inventory["status"] = "PASS"
        candidate_record = _json(output / "candidate-sweep-hot-question" / "result.json")
        observer = CampaignObserver(len(policies))
        qualifications = [{"name": row["name"], "status": "PASS" if row.get("exact_match") is True else "FAIL",
            "native_exit_code": candidate_record.get("native_exit_code"),
            "native_pid_absent": candidate_record.get("native_pid_absent"),
            "exact_match": row.get("exact_match"),
            "reference_tokens": row.get("reference", {}).get("tokens"),
            "candidate_tokens": row.get("actual", {}).get("tokens"),
            "reference_next_draw": row.get("reference", {}).get("next_draw"),
            "candidate_next_draw": row.get("actual", {}).get("next_draw")} for row in candidate_record.get("policy_results", [])]
        inventory["candidate_policy_qualification"] = observer.qualify(qualifications)
        if not inventory["candidate_policy_qualification"]:
            inventory["status"] = "FAIL"
            _write(output / "campaign.json", inventory)
            raise AssertionError("Candidate policy sweep did not meet every exact-match/native-exit gate")
        _write(output / "campaign.json", inventory)
        return

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"status": "RUNNING", "stage": args.stage, "python_pid": os.getpid(),
              "profile_sha256": _sha(args.profile), "fixture_sha256": _sha(args.fixture),
              "script_sha256": _sha(__file__), "failures": [], "native_exit_code": None,
              "native_pid_absent": False}
    _write(output / "result.json", result)
    raw = backend = None
    heartbeat = None
    try:
        sys.path[:0] = [str(runtime), str(runtime / "tools"),
                       str(runtime / "experimental/branching_context/src"),
                       str(runtime / "experimental/branching_context/scripts")]
        from grammar_long_reference import (NativePipe, Heartbeat, _native_environment, _apply_profile_environment,
            _sample_frontier, validate_feed_receipt, validate_decode_receipt)
        from context_time_travel.strata_backend import StrataBackend, _validate_execution, _validate_model
        from context_time_travel.real import BranchingModel
        from context_time_travel.sampled_controller import PlainSamplingSession, start
        from context_time_travel.store import _json_bytes
        from context_time_travel import min_p
        from tools.strata_tokenizer import Tokenizer
        heartbeat = Heartbeat(output / "heartbeat.json").start()
        _apply_profile_environment(profile)
        _validate_execution(profile)
        model_check = _validate_model(profile)
        fixture = _json(args.fixture)
        prefix = tuple(fixture["root_token_ids"][:args.prefix])
        branch = "A"
        suffix_all = fixture["branches"][branch]["suffix_token_ids"]
        suffix = tuple(suffix_all[-args.suffix:])
        if len(prefix) != args.prefix or len(suffix) != args.suffix:
            raise ValueError("Fixture does not contain requested prefix/suffix")
        tokenizer = Tokenizer.from_gguf(profile["model_path"])
        vocab = len(tokenizer.tokens)
        eos = tuple(profile["execution"]["eos_tokens"])
        mask = ((1 << vocab) - 1).to_bytes((vocab + 7) // 8, "little")
        result.update(model_verification=model_check, vocab_size=vocab, selected_branch=branch,
                      prefix_tokens=len(prefix), suffix_tokens=len(suffix), eos_tokens=list(eos))
        _write(output / "selected-fixture.json", {"fixture_sha256": _sha(args.fixture), "prefix": list(prefix),
               "suffix": list(suffix), "branch": branch, "selection": "first prefix tokens; final suffix tokens"})
        policy_name = args.policy_name
        switch_name = args.switch_policy_name
        if args.stage == "candidate-sweep":
            selected_policy = next(row["sampling"] for row in policies if row["name"] == policies[0]["name"])
        else:
            if not policy_name or policy_name not in names:
                raise ValueError("A declared --policy-name is required")
            selected_policy = next(row["sampling"] for row in policies if row["name"] == policy_name)
        switch_policy = None
        if args.stage.startswith("switch-"):
            if not switch_name or switch_name not in names:
                raise ValueError("Switch stages require a declared --switch-policy-name")
            switch_policy = next(row["sampling"] for row in policies if row["name"] == switch_name)
        result["policy_name"] = policy_name
        result["sampling"] = selected_policy
        result["sampling_protocol_expected"] = profile["execution"].get("sampling_protocol")
        samples = {}
        tokens = []

        if args.stage in ("reference", "switch-reference"):
            raw = NativePipe(profile["execution"]["sut_command"], _native_environment(profile), output)
            result["native_pid"] = raw.process.pid
            heartbeat.set("raw-reference-ready", raw.process.pid)
            if raw.startup_error:
                raise raw.startup_error
            if raw.ready.get("sampling_protocol") != "full-vocab-min-p-u53-v2" or raw.ready.get("vocab") != vocab:
                raise AssertionError("Native v2 sampler capability/vocabulary mismatch")
            position = 0
            for label, ids in (("prefix", prefix), ("question", suffix)):
                heartbeat.set("raw-feed-" + label, raw.process.pid)
                receipt = raw.request("feed " + " ".join(map(str, ids)), "feed")
                validate_feed_receipt(receipt, position=position, count=len(ids),
                                      chunk_size=profile["execution"]["prefill_chunk"])
                position += len(ids)
                samples[label] = _sample_frontier(raw, output, label, position, vocab)
            history = prefix + suffix
            session = PlainSamplingSession.restore(start(selected_policy, STREAM, history, vocab, eos), history,
                vocab, tokenizer.token_bytes, lambda token: token in eos)
            policy_for_token = selected_policy
            initial_next_draw = session.current_controller["next_draw"]
            switched = False
            for index in range(args.outputs):
                heartbeat.set("raw-sample-" + str(index + 1), raw.process.pid)
                if args.stage == "switch-reference" and index == args.switch_after and not switched:
                    session = PlainSamplingSession.restore(session.current_controller, history + tuple(tokens), vocab,
                        tokenizer.token_bytes, lambda token: token in eos)
                    session.configure_generation(switch_policy)
                    policy_for_token = switch_policy
                    initial_next_draw = session.current_controller["next_draw"]
                    switched = True
                logits_path = output / (("question" if index == 0 else "generated-" + str(index)) + ".f32")
                logits = struct.unpack("<" + "f" * vocab, logits_path.read_bytes())
                token = session.choose(mask, reference_selector(logits,policy_for_token))
                decode = raw.request("decode " + str(token), "decode")
                validate_decode_receipt(decode, position=position)
                position += 1
                session._consume_selected(token, tokenizer.token_bytes(token), token in eos)
                tokens.append(token)
                samples["generated-" + str(index + 1)] = _sample_frontier(raw, output,
                    "generated-" + str(index + 1), position, vocab)
                controller_bytes = _json_bytes(session.current_controller)
                if len(tokens) == args.switch_after:
                    (output / ("controller-" + str(len(tokens)) + ".json")).write_bytes(controller_bytes)
                _write(output / "result.json", {**result, "tokens": tokens, "samples": samples,
                       "controller_sha256": hashlib.sha256(controller_bytes).hexdigest(),
                       "controller_bytes": len(controller_bytes), "next_draw": session.current_controller["next_draw"]})
                if session.is_stopped():
                    break
            if args.stage == "switch-reference" and not switched:
                raise AssertionError("EOS ended the reference before the requested setting switch")
            stats = raw.request("stats", "stats")
            if stats.get("forward_tokens") != len(prefix) + len(suffix) + len(tokens) or stats.get("sampling_calls") != 0:
                raise AssertionError("Raw reference work counters differ from its explicit feed/decode schedule")
            final_controller_bytes = _json_bytes(session.current_controller)
            (output / "controller.json").write_bytes(final_controller_bytes)
            from grammar_long_reference import validate_command_inventory
            validate_command_inventory(raw.commands)
            result.update(tokens=tokens, samples=samples,
                          controller_sha256=hashlib.sha256(final_controller_bytes).hexdigest(),
                          controller_bytes=len(final_controller_bytes), next_draw=session.current_controller["next_draw"], stats=stats,
                          initial_next_draw=initial_next_draw, commands=raw.commands, exact_match=True, status="PASS")
        elif args.stage == "prepare":
            backend = StrataBackend(profile)
            result["native_pid"] = backend.process.pid
            heartbeat.set("prepare-prefix", backend.process.pid)
            model = BranchingModel(output / "store", backend, profile)
            root = model.create(prefix)
            question = model._feed(root, suffix, model.controller(root))
            from context_time_travel.sampled_controller import start as start_sampling
            sampled_state = start_sampling(selected_policy, STREAM, model.history(question), vocab, eos)
            question = model._controller_transition(question, sampled_state)
            prepared_samples={}
            for boundary,snapshot in (('prefix',root),('question',question)):
                values=model.cache[snapshot].frontier
                prepared_bytes=struct.pack('<'+'f'*vocab,*values)
                prepared_samples[boundary]=_frontier(prepared_bytes,output,boundary,vocab)
            for row in policies:
                ref=_json(output.parent/('reference-'+row['name'])/'result.json')
                for boundary in ('prefix','question'):
                    if prepared_samples[boundary]['sha256']!=ref['samples'][boundary]['sha256']:
                        raise AssertionError('Prepared '+boundary+' differs from fresh '+row['name'])
            result['samples']=prepared_samples
            head = model.store.fork("Question", root)
            model.save_branch(head.name, head.revision, question, "generation-settings-question")
            restore = model.restore_checkpoint(question)
            if restore.get("forward_tokens") != 0:
                raise AssertionError("Question checkpoint restore replayed prefix work")
            before_prepare = backend.work_counters()
            # Capture explicit prepare work before it is hidden by the subsequent restore receipt.
            if before_prepare.get("forward_tokens") != len(prefix) + len(suffix):
                raise AssertionError("Prepared root/question consumed an unexpected prefill schedule")
            controller_bytes = _json_bytes(model.controller(question))
            (output / "question-controller.json").write_bytes(controller_bytes)
            result.update(root=root, question=question, question_history_length=len(model.history(question)),
                          question_controller_sha256=hashlib.sha256(controller_bytes).hexdigest(),
                          question_controller_bytes=len(controller_bytes), prepare_counters=before_prepare,
                          restore_receipt=restore,
                          root_payload_sha256=backend.audit(model.cache[root].blob)["sha256"],
                          question_payload_sha256=backend.audit(model.cache[question].blob)["sha256"],
                          status="PASS")
            _write(output / "result.json", result)
        elif args.stage == "candidate-sweep":
            backend = StrataBackend(profile)
            result["native_pid"] = backend.process.pid
            heartbeat.set("candidate-restore", backend.process.pid)
            model = BranchingModel(output.parent / "prepare-shared" / "store", backend, profile)
            prep = _json(output.parent / "prepare-shared" / "result.json")
            question = prep["question"]
            before_restore = backend.work_counters()
            restore = model.restore_checkpoint(question)
            after_restore = backend.work_counters()
            if restore.get("forward_tokens") != 0 or after_restore["forward_tokens"] != before_restore["forward_tokens"]:
                raise AssertionError("Strict candidate restore performed forward work")
            source_image = model.cache[question].blob
            parent_hash = backend.audit(source_image)["sha256"]
            results = []
            for row in policies:
                heartbeat.set("candidate-" + row["name"], backend.process.pid)
                policy_output = output / row["name"]
                policy_output.mkdir(mode=0o700, exist_ok=False)
                reference = _json(output.parent / ("reference-" + row["name"]) / "result.json")
                capture = _candidate_request(model, backend, question, row["sampling"], tokenizer, vocab,
                    policy_output, args.outputs, reference,
                    progress=lambda count, name=row["name"]: heartbeat.set("candidate-" + name + "-token-" + str(count), backend.process.pid))
                if capture["counters_before"].get("loaded_file_bytes") != capture["counters_after"].get("loaded_file_bytes"):
                    raise AssertionError("A policy after the strict restore performed another disk load")
                results.append(dict(name=row["name"], sampling=row["sampling"], **capture,
                    exact_match=True, work_counters_before=capture["counters_before"],
                    work_counters_after=capture["counters_after"]))
                result["policy_results"] = results
                _write(output / "result.json", result)
            if backend.audit(source_image)["sha256"] != parent_hash:
                raise AssertionError("Question source payload changed during policy sweep")
            result.update(restore_receipt=restore, restore_counters_before=before_restore,
                          restore_counters_after=after_restore, parent_payload_sha256=parent_hash,
                          exact_match=True, status="PASS")
            _write(output / "result.json", result)
        elif args.stage == "switch-prepare":
            backend = StrataBackend(profile)
            result["native_pid"] = backend.process.pid
            store_path = output.parent / "prepare-shared" / "store"
            model = BranchingModel(store_path, backend, profile)
            question = _json(output.parent / "prepare-shared" / "result.json")["question"]
            restore = model.restore_checkpoint(question)
            if restore.get("forward_tokens") != 0:
                raise AssertionError("Switch setup restore replayed prefix work")
            heartbeat.set("switch-prefix", backend.process.pid)
            reference = _json(output.parent / "switch-reference-switch-case" / "result.json")
            prepared_samples=_json(output.parent/'prepare-shared/result.json')['samples']
            for boundary in ('prefix','question'):
                if prepared_samples[boundary]['sha256']!=reference['samples'][boundary]['sha256']:
                    raise AssertionError('Switched reference differs from measured prepared '+boundary)
            source_before = backend.audit(model.cache[question].blob)["sha256"]
            partial_capture = _candidate_request(model, backend, question, selected_policy, tokenizer,
                vocab, output, args.switch_after, reference, compare=False, expected_initial_draw=0,
                progress=lambda count: heartbeat.set("switch-prefix-token-" + str(count), backend.process.pid))
            partial = partial_capture["answer"]
            if partial["tokens"] != reference["tokens"][:args.switch_after]:
                raise AssertionError("Pre-restart switch tokens differ from raw reference")
            expected_partial_samples = {"question": reference["samples"]["question"]["sha256"]}
            expected_partial_samples.update({"generated-" + str(index + 1):
                reference["samples"]["generated-" + str(index + 1)]["sha256"]
                for index in range(args.switch_after)})
            if partial_capture["actual"]["samples"] != expected_partial_samples:
                raise AssertionError("Pre-restart switch frontier differs from raw reference")
            if (partial_capture["actual"]["controller_sha256"] !=
                    _sha(output.parent / "switch-reference-switch-case" / ("controller-" + str(args.switch_after) + ".json"))):
                raise AssertionError("Pre-restart switch controller differs from raw reference")
            reference_partial_controller = _json(output.parent / "switch-reference-switch-case" / ("controller-" + str(args.switch_after) + ".json"))
            partial_next_draw = partial_capture["actual"].get("next_draw")
            if (type(partial_next_draw) is not int
                    or partial_next_draw != reference_partial_controller.get("next_draw")):
                raise AssertionError("Saved partial draw cursor differs from the raw reference boundary")
            head = model.store.fork("SwitchedPartial", partial["snapshot"])
            model.save_branch(head.name, head.revision, partial["snapshot"], "sampler-switch-partial")
            if backend.audit(model.cache[question].blob)["sha256"] != source_before:
                raise AssertionError("Question parent payload changed during switched partial generation")
            result.update(partial_capture, question=question, partial_snapshot=partial["snapshot"], partial_tokens=partial["tokens"],
                          partial_controller_sha256=partial_capture["actual"]["controller_sha256"],
                          partial_next_draw=partial_next_draw,
                          history=model.history(partial["snapshot"]),
                          question_payload_sha256=source_before, counters=backend.work_counters(), status="PASS")
        elif args.stage == "switch-resume":
            backend = StrataBackend(profile)
            result["native_pid"] = backend.process.pid
            model = BranchingModel(output.parent / "prepare-shared" / "store", backend, profile)
            heartbeat.set("switch-resume-restore", backend.process.pid)
            prepared = _json(output.parent / "switch-prepare-switch-case" / "result.json")
            restore = model.restore_checkpoint(prepared["partial_snapshot"])
            before = backend.work_counters()
            if restore.get("forward_tokens") != 0:
                raise AssertionError("Switched checkpoint restore replayed prefix work")
            partial_image = model.cache[prepared["partial_snapshot"]].blob
            partial_hash = backend.audit(partial_image)["sha256"]
            reference = _json(output.parent / "switch-reference-switch-case" / "result.json")
            restored_frontier=struct.pack('<'+'f'*vocab,*model.cache[prepared['partial_snapshot']].frontier)
            restored_sha256=hashlib.sha256(restored_frontier).hexdigest()
            if restored_sha256 != reference['samples']['generated-'+str(args.switch_after)]['sha256']:
                raise AssertionError('Restarted initial frontier differs from raw reference at the settings boundary')
            result['restored_frontier_sha256']=restored_sha256
            if (type(prepared.get("partial_next_draw")) is not int
                    or prepared["partial_next_draw"] != reference.get("initial_next_draw")):
                raise AssertionError("Saved partial draw cursor differs from switched raw reference base")
            capture = _candidate_request(model, backend, prepared["partial_snapshot"], switch_policy, tokenizer,
                                         vocab, output, args.outputs - args.switch_after, reference,
                                         expected_prefix=prepared["partial_tokens"],
                                         expected_question_sha256=prepared['actual']['samples']['question'],
                                         expected_prefix_samples={key: value["sha256"] for key, value in prepared["samples"].items()},
                                         expected_initial_draw=prepared["partial_next_draw"],
                                         progress=lambda count: heartbeat.set("switch-resume-token-" + str(count), backend.process.pid))
            if backend.audit(partial_image)["sha256"] != partial_hash:
                raise AssertionError("Saved partial source payload changed during restarted generation")
            result.update(capture, restore_receipt=restore, restore_counters_before=before,
                          partial_payload_sha256=partial_hash, restore_counters_after=backend.work_counters(),
                          exact_match=True, status="PASS")
    except BaseException as error:
        result["status"] = "FAIL"
        result["failures"].append({"type": type(error).__name__, "message": str(error)[:500]})
        raise
    finally:
        if backend is not None:
            from run_grammar_first_probe import finish_backend
            finish_backend(backend, result)
        if raw is not None:
            try:
                code, absent, errors = raw.close()
                result.update(native_exit_code=code, native_pid_absent=absent, native_cleanup_errors=errors)
                if type(code) is not int or code != 0 or absent is not True or errors:
                    result["status"] = "FAIL"
            except BaseException as error:
                result["native_cleanup_error"] = type(error).__name__
                result["status"] = "FAIL"
        if heartbeat is not None:
            heartbeat.close()
        _write(output / "result.json", result)

    if result['status']!='PASS':
        raise AssertionError('Stage failed, including process cleanup')


def _candidate_request(model, backend, snapshot, policy, tokenizer, vocab, output, count, reference,
                       expected_prefix=(), expected_prefix_samples=None, compare=True, progress=None,
                       expected_initial_draw=None, expected_question_sha256=None):
    samples = {}
    original_generated = backend.consume_generated
    generated = 0
    def capture(lease, ids):
        nonlocal generated
        original_generated(lease, ids)
        generated += len(ids)
        if progress is not None:
            progress(generated)
        values = backend.logits(lease)
        raw = struct.pack("<" + "f" * vocab, *values)
        samples["generated-" + str(generated)] = _frontier(
            raw, output, "candidate-generated-" + str(generated), vocab)
    backend.consume_generated = capture
    before = backend.work_counters()
    try:
        answer = model.run_sampled_request(snapshot, (), count, sampling=policy, stream=STREAM, resume=True)
    finally:
        backend.consume_generated = original_generated
    after = backend.work_counters()
    expected_forward = len(answer["tokens"])
    mask_bytes = (vocab + 7) // 8
    for key, expected in (("forward_tokens", expected_forward), ("prefill_tokens", 0),
                          ("decode_timed_tokens", expected_forward), ("sampling_calls", expected_forward),
                          ("sampling_logit_bytes", vocab * 4 * expected_forward),
                          ("sampling_mask_bytes", mask_bytes * expected_forward)):
        if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key] - before[key] != expected:
            raise AssertionError("Unexpected candidate work delta: " + key)
    controller = model.controller(answer["snapshot"])
    from context_time_travel.store import _json_bytes
    controller_bytes = _json_bytes(controller)
    controller_path = Path(output) / "controller.json"
    controller_path.write_bytes(controller_bytes)
    controller_hash = hashlib.sha256(controller_bytes).hexdigest()
    expected_draws = 0 if policy["temperature"] == 0 else len(answer["tokens"])
    initial_draw = reference.get("initial_next_draw", 0) if expected_initial_draw is None else expected_initial_draw
    if controller.get("next_draw") != expected_draws + initial_draw:
        raise AssertionError("Candidate RNG draw cursor differs from the policy draw contract")
    if compare and controller_hash != reference.get("controller_sha256"):
        raise AssertionError("Candidate final controller differs from fresh reference")
    if model.history(answer["snapshot"]) != model.history(snapshot) + tuple(answer["tokens"]):
        raise AssertionError("Candidate canonical history differs from exact generated tokens")
    reference_samples = {key: value["sha256"] if isinstance(value, dict) else value
                         for key, value in continuation_samples(reference.get("samples", {})).items()}
    actual_samples = {key: row["sha256"] for key, row in samples.items()}
    if expected_prefix:
        if (not isinstance(expected_question_sha256,str) or len(expected_question_sha256)!=64
                or any(ch not in '0123456789abcdef' for ch in expected_question_sha256)):
            raise AssertionError('Restart join requires the actually measured original question frontier')
        if (not isinstance(expected_prefix_samples, dict)
                or set(expected_prefix_samples) != {"generated-" + str(i + 1) for i in range(len(expected_prefix))}):
            raise AssertionError("Restarted continuation lacks measured pre-restart frontier hashes")
        actual_samples = {"question": expected_question_sha256} | dict(expected_prefix_samples) | {
            "generated-" + str(int(key.split("-")[-1]) + len(expected_prefix)): value
            for key, value in actual_samples.items()}
    else:
        question_frontier = struct.pack("<" + "f" * vocab, *model.cache[snapshot].frontier)
        actual_samples["question"] = hashlib.sha256(question_frontier).hexdigest()
    actual = {"tokens": list(expected_prefix) + answer["tokens"], "controller_sha256": controller_hash,
              "next_draw": controller.get("next_draw"),
              "controller_bytes": len(controller_bytes),
              "samples": actual_samples}
    if compare:
        _equal(actual["tokens"], reference["tokens"], "tokens")
        if controller_hash != reference.get("controller_sha256"):
            raise AssertionError("Exact same-schedule mismatch: controller")
        _equal(actual["samples"], reference_samples, "frontiers")
    return {"actual": actual, "reference": reference, "answer": answer, "samples": samples,
            "counters_before": before, "counters_after": after}


if __name__ == "__main__":
    main()
