#!/usr/bin/env python3
"""Derive a bounded three-context browser fixture from hashed source receipts.

This is source selection only. The separate tokenizer/template fixture builder
must validate the resulting root and question token counts against the chosen
execution profile before a browser run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


QUESTIONS = {
    'native_cow_engine': (
        "Using only the supplied native engine source, explain how a saved conversation is captured and then "
        "forked without changing its parent. Trace the real call path from the public session operation through "
        "the page table, mapping ownership, first divergent write, and snapshot cleanup. Identify the exact "
        "objects that are shared, the point where a private page is allocated or copied, and the checks that "
        "prevent a failed capture or fork from publishing a partially valid image. Separate source-supported "
        "facts from assumptions that would require a running CUDA test. Do not infer that a pointer or checksum "
        "proves physical GPU residency.\n\n"
        "Then review the same code as a failure analysis. What happens if page allocation, mapping, asynchronous "
        "copy, digest calculation, or synchronization fails? Which resources are retained by the parent, which "
        "temporary resources must be released, and how does the API prevent later inference from using a "
        "partially applied child? Cite method names and describe the order of operations. If the code does not "
        "establish a guarantee, say what additional test or runtime observation would be needed. Avoid treating "
        "comments as proof when the implementation follows a different path.\n\n"
        "Organize the review around the actual ownership transitions visible in these excerpts. For each stage, "
        "state the input invariant, the mutation boundary, and the cleanup responsibility; distinguish a logical "
        "page-table change from a device allocation or peer mapping. Explain how a reader could tell whether a "
        "snapshot still refers to shared parent pages after the child diverges, and what source evidence is "
        "insufficient to establish that fact at runtime. Include one concrete failure sequence and trace which "
        "objects should remain usable afterward. Where a helper or call path is outside the supplied excerpt, "
        "identify the missing source instead of assuming its behavior.\n\n"
        "Finish with a compact test plan tied to these methods: name a parent/child byte comparison, a failed "
        "mapping or copy case, and a repeated capture/load case. For each, say what exact observation would "
        "support the invariant and what result would falsify it. Keep expected behavior separate from tests that "
        "have actually run, and do not claim that CPU-side structure inspection proves CUDA execution."),
    'canonical_api_store': (
        "Using only the supplied canonical store and coordinator source, trace a branch fork from its selected "
        "source branch and captured revision through validation, intent persistence, checkpoint retention, "
        "publication, and retry recovery. Explain how the source snapshot and presentation namespace are "
        "bound, how the operation identifier prevents changed-input retries, and why a fork must not infer "
        "which chat owns a shared canonical root. Name the APIs that validate ancestry and controller state. "
        "Do not infer GPU work from a durable HTTP receipt.\n\n"
        "Then review the read and failure paths. Explain how history and presentation records are checked "
        "against canonical token spans, how corrupt or missing metadata fails closed, and what remains "
        "authoritative after a rewind. Distinguish durable branch heads from derived message projections and "
        "from RAM or disk checkpoint residency. Identify which cleanup runs after a failed publication and "
        "which state must remain available for an exact retry. Support each conclusion with method names and "
        "state fields, and label any claim that would still need an executed concurrency or persistence test.\n\n"
        "Trace one concrete fork, move, retry, and rewind using the fields present in the excerpt. State which "
        "record is immutable, which record is the current head, and how a reader can distinguish a captured "
        "source revision from a later source head. Explain why equal token histories do not by themselves identify "
        "the same chat or authorize merging presentation records. For each validation step, name the data it "
        "checks and the failure that should result if it disagrees. If a digest protects serialized bytes, do not "
        "treat it as a substitute for namespace, ancestry, or renderer validation.\n\n"
        "End by proposing exact regression assertions for changed-input retry rejection, corrupt presentation "
        "metadata, and a rewind that preserves older receipts without showing them on the active path. Separate "
        "those source-derived expectations from executed evidence, and identify any behavior that requires the "
        "coordinator or another file not included in the selected excerpt."),
    'serving_tooling_template': (
        "Using only the supplied serving and template source, trace how an uploaded text file and a user "
        "question become a canonical native request. Explain file framing, role assignment, template "
        "rendering, tokenization, the context-only root boundary, and how ask or append-user differs from "
        "prepare-and-answer. Identify the checks that keep file content in the user role and prevent literal "
        "template tags from becoming instructions. Name the request and durable acknowledgement boundaries; "
        "do not claim that a browser stream alone proves a saved checkpoint.\n\n"
        "Then inspect the client and server failure behavior. Explain how operation IDs and idempotent retries "
        "preserve drafts, how incomplete or truncated streams are distinguished from durable completion, and "
        "how a graph action maps to an actual saved snapshot rather than an invented per-message offset. "
        "Separate source-level behavior from claims requiring a real browser or GPU run. Cite the relevant "
        "renderer, route, or UI methods, and describe what evidence an auditor should compare after the "
        "request completes.\n\n"
        "Walk through the file-only context path and the later question path as two distinct operations. Name "
        "the fields that carry the attachment, role, renderer identity, namespace, branch, and source snapshot. "
        "Explain what the source proves about literal template markers inside uploaded text and what must be "
        "checked in rendered bytes or token IDs. Identify where a request can be accepted but not yet durably "
        "published, and how the client should preserve the user's question if that boundary is interrupted. "
        "Do not equate a visible streamed answer with a completed stored message.\n\n"
        "Conclude with a small falsifiable browser/server test: create a context-only root, ask two independent "
        "questions from that root, and compare the durable source snapshot and branch histories. State which "
        "request bodies, operation receipts, and final snapshots the auditor should retain. If a UI event or "
        "server response alone cannot prove a property, say which canonical store record or native counter "
        "would be needed."),
}

QUESTIONS_B = {
    'native_cow_engine': (
        "For an independent review of the supplied engine source, focus on the saved image and device boundary. "
        "Describe the metadata, geometry, token history, payload sizes, and frontier values that must be checked "
        "before any private GPU page is prepared. Explain how the streaming reader binds validation and apply "
        "to one opened file, how it detects in-place changes, and why an atomic rename may safely leave the old "
        "opened inode in use. Identify the digest algorithm and do not call a non-cryptographic checksum a "
        "cryptographic proof.\n\n"
        "Then analyze failure containment. If preparation succeeds but a later read, sink, digest, copy, or "
        "synchronization step fails, what state must become unusable, what must not be published, and how are "
        "the parent and an existing sibling kept unchanged? Name the relevant cleanup boundaries and distinguish "
        "a parser-only check from a CUDA execution test. State what evidence would be needed to claim that a "
        "partial device apply cannot be followed by inference, and identify any remaining ambiguity in the code.\n\n"
        "Organize this review by validation order: first the serialized header and geometry, then token history, "
        "payload layout and digest, then frontier and private destination preparation. Explain why the first "
        "pass must establish limits before allocation and why the second pass must use the same opened file "
        "description rather than reopening a replaceable pathname. Describe which metadata must be compared "
        "again during apply and what a mismatch means for the session. State clearly which checksum properties "
        "are guaranteed by the implementation and which would require an outer cryptographic digest.\n\n"
        "Finish with three concrete tests: corrupt or truncate bytes before preparation and show no destination "
        "write; replace the pathname with a same-sized valid file between passes and show that the validated "
        "opened file remains the input; then inject a post-prepare read or device-copy failure and show the "
        "session cannot be used or published while the frozen parent and sibling remain unchanged. Keep a "
        "parser test distinct from an actual CUDA test, and do not claim a test ran unless the evidence says so."),
    'canonical_api_store': (
        "For an independent review of the supplied store and coordinator source, focus on rewind and resumed "
        "answers. Explain which immutable receipts remain after a branch moves backward, how the current head "
        "is distinguished from a retained prior tip, and how an assistant message ID continues across resume "
        "without merging divergent sibling text. Describe the provenance required to fork from a captured "
        "older branch revision and what a retry must match before returning the original result.\n\n"
        "Then inspect the validation boundary. Which checks bind token history, controller state, namespace, "
        "presentation spans, and source ancestry? Explain what the read-only projection may derive from those "
        "records and what it must not create as another head or history. Identify how corruption, missing "
        "references, mismatched operation input, and an unauthorized principal fail. Distinguish a durable "
        "metadata receipt from a loaded checkpoint or a successfully resumed native worker, and state which "
        "claims need a separate concurrency or process-restart test.\n\n"
        "Trace one concrete fork, move, retry, and rewind using the fields present in the excerpt. State which "
        "record is immutable, which record is the current head, and how a reader can distinguish a captured "
        "source revision from a later source head. Explain why equal token histories do not by themselves identify "
        "the same chat or authorize merging presentation records. For each validation step, name the data it "
        "checks and the failure that should result if it disagrees. If a digest protects serialized bytes, do not "
        "treat it as a substitute for namespace, ancestry, or renderer validation.\n\n"
        "End by proposing exact regression assertions for changed-input retry rejection, corrupt presentation "
        "metadata, and a rewind that preserves older receipts without showing them on the active path. Separate "
        "those source-derived expectations from executed evidence, and identify any behavior that requires the "
        "coordinator or another file not included in the selected excerpt."),
    'serving_tooling_template': (
        "For an independent review of the supplied serving and template source, focus on append boundaries. "
        "Explain how the encoder recognizes a closed user message, how an active assistant differs from a "
        "stopped turn, and why the appended suffix must come from the pinned renderer instead of a guessed token "
        "count. Trace how branch identity, operation ID, source snapshot, and user text travel from the browser "
        "to the server and how the client decides that a complete event is durable.\n\n"
        "Then review adversarial cases: a truncated stream, a lost acknowledgement after publication, a stale "
        "graph boundary, a sibling branch selected in the UI, and file content containing literal template "
        "tags. State which existing checks prevent each mix-up and what they do not prove. Identify the "
        "smallest browser or server test that would expose any remaining ambiguity. Keep user text in the "
        "user role, distinguish provisional stream bytes from a canonical saved message, and do not treat a "
        "successful HTTP response as evidence of GPU residency or checkpoint reuse.\n\n"
        "Walk through the file-only context path and the later question path as two distinct operations. Name "
        "the fields that carry the attachment, role, renderer identity, namespace, branch, and source snapshot. "
        "Explain what the source proves about literal template markers inside uploaded text and what must be "
        "checked in rendered bytes or token IDs. Identify where a request can be accepted but not yet durably "
        "published, and how the client should preserve the user's question if that boundary is interrupted. "
        "Do not equate a visible streamed answer with a completed stored message.\n\n"
        "Conclude with a small falsifiable browser/server test: create a context-only root, ask two independent "
        "questions from that root, and compare the durable source snapshot and branch histories. State which "
        "request bodies, operation receipts, and final snapshots the auditor should retain. If a UI event or "
        "server response alone cannot prove a property, say which canonical store record or native counter "
        "would be needed."),
}

QUESTION_EXPANSIONS = {
    'native_cow_engine': (
        "For every claimed guarantee, distinguish a source-level precondition from a postcondition that a "
        "test must observe. Explain how parent ownership, child ownership, mapping lifetime, and cleanup "
        "interact when the first write occurs after a snapshot. If the code uses counters, name what they count "
        "and avoid interpreting a copy count as a byte-for-byte equality result. Include the exact additional "
        "instrumentation needed to attribute a failed operation to allocation, transfer, digest, or synchronization."
    ),
    'canonical_api_store': (
        "For each transition, distinguish the durable event from its derived display projection and from the "
        "checkpoint's physical residency. Explain which identifiers must remain stable across retry and which "
        "revision changes on a successful move. Describe how you would detect an event that is valid JSON but "
        "references an absent or unrelated snapshot. Include the expected behavior for two namespaces whose "
        "canonical roots happen to contain identical token IDs, and identify which explicit metadata prevents "
        "their histories from being combined."
    ),
    'serving_tooling_template': (
        "For each step, distinguish browser intent, server acceptance, native work, and durable publication. "
        "Explain what exact request and response fields let an auditor join those stages without relying on "
        "display text or event timing. Include an interrupted-request case where publication may have happened "
        "before the client lost its acknowledgement, and state how the operation ID and retained draft should "
        "be used to recover without submitting changed text. Keep all file text untrusted and in its user-facing "
        "data path throughout the explanation."
    ),
}

QUESTION_B_EXPANSIONS = {
    'native_cow_engine': (
        "Conclude by separating immutable image data, mutable session state, and device mappings in your "
        "proposed audit. State how a test should compare the entire parent and sibling payload before and after "
        "a child failure, not just a marker or one selected byte. Explain what a clean process restart proves "
        "about serialization and what it still cannot prove about numerical model behavior. If the shown source "
        "does not expose a required counter or failure hook, name that limitation explicitly."
    ),
    'canonical_api_store': (
        "Conclude with a concrete adversarial example in which two presentations share canonical text but have "
        "different namespace or fork provenance. State which one must appear in a selected branch projection and "
        "which records remain available as saved points after rewind. Explain what a CPU-only store test can "
        "establish, what requires a browser request, and what must be checked against an independently encoded "
        "token history. Do not turn a matching text body into an identity rule."
    ),
    'serving_tooling_template': (
        "Conclude with exact evidence for both branches: the original context snapshot, each submitted request "
        "body and operation ID, the durable completion record, and the resulting canonical history. Explain how "
        "to tell an active bounded assistant from a completed answer, and why a later resume extends the same "
        "logical message instead of adding another user-visible answer. Mark source-review expectations and "
        "executed browser observations as separate evidence classes."
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(source_manifest: Path, output: Path, *, target_tokens: int = 7800) -> dict:
    if type(target_tokens) is not int or not 6000 <= target_tokens <= 9000:
        raise ValueError('Target source tokens must be between 6000 and 9000')
    manifest_bytes = source_manifest.read_bytes()
    source = json.loads(manifest_bytes)
    contexts = source.get('contexts')
    if not isinstance(contexts, list) or len(contexts) != 3:
        raise ValueError('Source evidence must contain exactly three contexts')
    names = [row.get('name') for row in contexts if isinstance(row, dict)]
    if set(names) != set(QUESTIONS) or len(names) != 3:
        raise ValueError('Source evidence must match the three supported source domains')
    output = output.resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    result_contexts = []
    used_body_hashes = set()
    for context in contexts:
        name = context['name']
        original = context.get('root_messages')
        receipts = context.get('included_files_in_order')
        original_tokens = context.get('base_token_count')
        if (not isinstance(original, list) or not original or not isinstance(original[-1], dict)
                or not isinstance(original[-1].get('content'), str)
                or type(original_tokens) is not int or original_tokens <= 0
                or not isinstance(receipts, list) or not receipts):
            raise ValueError(f'{name}: source evidence lacks body, token count, or file receipts')
        target_chars = max(1, math.ceil(len(original[-1]['content']) * target_tokens / original_tokens))
        pieces, selected, used_chars = [], [], 0
        seen_paths = set()
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise ValueError(f'{name}: malformed source receipt')
            path_text = receipt.get('path')
            relative = receipt.get('relative_path')
            root_text = receipt.get('source_root')
            expected_sha = receipt.get('sha256')
            expected_bytes = receipt.get('bytes')
            if (not all(isinstance(value, str) and value for value in
                        (path_text, relative, root_text, expected_sha))
                    or type(expected_bytes) is not int or expected_bytes < 0):
                raise ValueError(f'{name}: incomplete source file receipt')
            path = Path(path_text)
            root = Path(root_text)
            if path.resolve() != (root / relative).resolve() or path.resolve() in seen_paths:
                raise ValueError(f'{name}: receipt path is inconsistent or duplicated')
            seen_paths.add(path.resolve())
            raw = path.read_bytes()
            if len(raw) != expected_bytes or _sha(raw) != expected_sha:
                raise ValueError(f'{name}: source file changed from its recorded SHA/size: {relative}')
            text = raw.decode('utf-8')
            lines = text.splitlines(keepends=True)
            header = f'SOURCE FILE: {relative}\n'
            remaining = target_chars - used_chars - len(header)
            if remaining <= 0:
                break
            excerpt_lines, excerpt_chars = [], 0
            for line in lines:
                if excerpt_chars >= remaining:
                    break
                excerpt_lines.append(line)
                excerpt_chars += len(line)
            if not excerpt_lines:
                continue
            excerpt = ''.join(excerpt_lines)
            excerpt_bytes = excerpt.encode('utf-8')
            first_line = 1
            last_line = len(excerpt_lines)
            byte_end = len(excerpt_bytes)
            pieces.extend((header, excerpt))
            used_chars += len(header) + len(excerpt)
            selected.append({'path': path_text, 'source_root': root_text,
                'relative_path': relative, 'source_file_bytes': len(raw),
                'source_file_sha256': expected_sha, 'line_start_1based': first_line,
                'line_end_1based': last_line, 'byte_start': 0, 'byte_end': byte_end,
                'excerpt_bytes': byte_end, 'excerpt_sha256': _sha(excerpt_bytes)})
            if used_chars >= target_chars:
                break
        body = ''.join(pieces)
        body_hash = _sha(body.encode('utf-8'))
        if not selected or body_hash in used_body_hashes:
            raise ValueError(f'{name}: bounded source excerpt is empty or duplicated')
        used_body_hashes.add(body_hash)
        root_messages = [{'role': 'user', 'content': body}]
        qa = QUESTIONS[name] + '\n\n' + QUESTION_EXPANSIONS[name]
        qb = QUESTIONS_B[name] + '\n\n' + QUESTION_B_EXPANSIONS[name]
        result_contexts.append({'name': name, 'root_messages': root_messages,
            'A_messages': root_messages + [{'role': 'user', 'content': qa}],
            'B_messages': root_messages + [{'role': 'user', 'content': qb}],
            'source_receipts': selected, 'source_body_sha256': body_hash,
            'source_body_chars': len(body), 'target_source_chars': target_chars,
            'source_token_target': target_tokens,
            'question_source': 'domain-specific authored review prompts; exact Q4 token counts are emitted later',
            'question_sha256': {'A': _sha(qa.encode()), 'B': _sha(qb.encode())}})
    if len(used_body_hashes) != 3:
        raise ValueError('Derived source bodies are not pairwise distinct')
    derived = {'schema': 'cow-browser-bounded-source-set-v1',
        'status': 'PASS_SOURCE_HASHED_EXCERPTS_TOKEN_COUNTS_PENDING',
        'source_evidence_path': str(source_manifest.resolve()),
        'source_evidence_sha256': _sha(manifest_bytes),
        'selection': {'method': 'ordered unique source files, whole-line prefixes, no padding or repeated text',
                      'target_source_tokens': target_tokens},
        'contexts': result_contexts}
    output.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8', newline='\n')
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-source-tokens', type=int, default=7800)
    args = parser.parse_args()
    result = derive(args.source_evidence, args.output, target_tokens=args.target_source_tokens)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
        'source_evidence_sha256': result['source_evidence_sha256'],
        'contexts': [{'name': row['name'], 'source_chars': row['source_body_chars'],
                      'source_body_sha256': row['source_body_sha256'],
                      'source_ranges': row['source_receipts']} for row in result['contexts']]},
        ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
