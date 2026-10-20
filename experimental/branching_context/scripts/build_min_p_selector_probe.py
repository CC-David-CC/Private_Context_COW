#!/usr/bin/env python3
"""Build an isolated full-vocabulary Min-P command from a frozen selector receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROTOCOL = "full-vocab-min-p-u53-v1"
PROTOCOL_V2 = "full-vocab-min-p-u53-v2"
PROTOCOL_V3 = "full-vocab-dry-min-p-u53-v3"
PROTOCOL_V4 = "full-vocab-typical-dyna-min-p-u53-v4"
PROTOCOL_V5 = "full-vocab-xtc-min-p-u53-v5"
PROTOCOL_V6 = "full-vocab-mirostat2-u53-v6"
PROTOCOL_V7 = "full-vocab-adaptive-p-u53-v7"
READY_ANCHOR = ',\\"grammar_selection_protocol\\":\\"finite-mask-argmax-v1\\"}'
STATS_ANCHOR = r'<< ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes'
COUNTERS_ANCHOR = ("uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, "
                   "grammar_selection_mask_bytes = 0;")


def _append_v5_response_receipt(command: str) -> str:
    """Append XTC v5 metadata, including the validated categorical mass."""
    anchor = r'<< ",\"effective_temperature\":" << selection.base.effective_temperature << "}";'
    replacement = (
        r'<< ",\"effective_temperature\":" << selection.base.effective_temperature'
        r' << ",\"total_weight\":" << selection.base.total_weight'
        r' << ",\"draw53_second\":" << selection.draw53_second'
        r' << ",\"draw_count\":" << selection.draw_count'
        r' << ",\"activation_draws\":" << selection.activation_draws'
        r' << ",\"activated\":" << (selection.activated ? "true" : "false")'
        r' << ",\"removed_count\":" << selection.removed_count'
        r' << ",\"pre_xtc_count\":" << selection.pre_xtc_count'
        r' << ",\"xtc_probability\":" << xtc_probability'
        r' << ",\"xtc_threshold\":" << xtc_threshold << "}";')
    if command.count(anchor) != 1:
        raise ValueError("v5 response anchor must occur exactly once")
    command = command.replace(anchor, replacement, 1)
    if command.count(r'<< ",\"total_weight\":" << selection.base.total_weight') != 1:
        raise ValueError("v5 total_weight receipt expression must occur exactly once")
    if command.count(r'<< ",\"draw53\":" << draw53') != 1:
        raise ValueError("v5 first draw receipt must occur exactly once")
    if command.count(r'<< ",\"draw53_second\":" << selection.draw53_second') != 1:
        raise ValueError("v5 second draw receipt must occur exactly once")
    return command


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def instrument_min_p_selector(source: str) -> str:
    """Insert one opt-in Min-P command into the generated selector driver."""
    if source.count(READY_ANCHOR) != 1:
        raise ValueError("grammar selector ready capability anchor must occur exactly once")
    if source.count(STATS_ANCHOR) != 1:
        raise ValueError("grammar selector stats anchor must occur exactly once")
    if source.count(COUNTERS_ANCHOR) != 1:
        raise ValueError("grammar selector counter declaration anchor must occur exactly once")
    if 'command == "sample_mask"' in source or "sampling_protocol" in source:
        raise ValueError("driver already contains a Min-P sampling protocol")
    include_anchor = '#include "grammar_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("grammar selector include anchor must occur exactly once")
    command_anchor = '} else if (command == "generate") {'
    if source.count(command_anchor) != 1:
        raise ValueError("generate command anchor must occur exactly once")

    ready = READY_ANCHOR[:-1] + ',\\"sampling_protocol\\":\\"' + PROTOCOL + '\\"}'
    source = source.replace(READY_ANCHOR, ready, 1)
    source = source.replace(include_anchor, include_anchor + '\n#include "min_p_select.h"', 1)
    source = source.replace(COUNTERS_ANCHOR,
        COUNTERS_ANCHOR[:-1] + ', sampling_calls = 0, sampling_logit_bytes = 0, '
        'sampling_mask_bytes = 0, sampling_ns = 0;', 1)
    command = r'''        } else if (command == "sample_mask") {
            int64_t expected_consumed = 0;
            double temperature = 0.0, min_p = 0.0;
            uint64_t draw53 = 0;
            std::string mask_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> draw53 >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("sample_mask requires consumed, temperature, min_p, draw53, and quoted mask path"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                draw53 >= (uint64_t(1) << 53)) {
                error_reply("sample_mask parameters are outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask mask read failed"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                                                 temperature, min_p, draw53);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            sampling_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            ++sampling_calls;
            sampling_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_mask_bytes += selection.mask_bytes;
            std::cout << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                      << "\"protocol\":\"full-vocab-min-p-u53-v1\",\"token\":" << selection.token
                      << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                      << ",\"mask_bytes\":" << selection.mask_bytes
                      << ",\"legal_count\":" << selection.legal_count
                      << ",\"kept_count\":" << selection.kept_count
                      << ",\"draw53\":" << draw53 << "}" << std::endl;
        } else if (command == "generate") {'''
    source = source.replace(command_anchor, command, 1)
    stats_new = STATS_ANCHOR \
        + r' << ",\"sampling_calls\":" << sampling_calls' \
        + r' << ",\"sampling_logit_bytes\":" << sampling_logit_bytes' \
        + r' << ",\"sampling_mask_bytes\":" << sampling_mask_bytes' \
        + r' << ",\"sampling_ns\":" << sampling_ns'
    source = source.replace(STATS_ANCHOR, stats_new, 1)
    stats_fields = (
        r' << ",\"sampling_calls\":" << sampling_calls',
        r' << ",\"sampling_logit_bytes\":" << sampling_logit_bytes',
        r' << ",\"sampling_mask_bytes\":" << sampling_mask_bytes',
        r' << ",\"sampling_ns\":" << sampling_ns',
    )
    if (source.count('command == "sample_mask"') != 1 or
            source.count("sampling_protocol") != 1 or
            source.count(STATS_ANCHOR) != 1 or
            any(source.count(field) != 1 for field in stats_fields)):
        raise ValueError("Min-P transform anchors are not unique: " + repr((
            source.count('command == "sample_mask"'), source.count("sampling_protocol"),
            tuple(source.count(field) for field in stats_fields))))
    return source


def instrument_min_p_selector_v2(source: str) -> str:
    """Add the versioned Min-P/Top-K/Top-P command while retaining v1."""
    source = instrument_min_p_selector(source)
    include_anchor = '#include "min_p_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("v2 selector include anchor must occur exactly once")
    source = source.replace(include_anchor,
        include_anchor + "\n#include <charconv>\n#include <limits>", 1)
    capability = 'sampling_protocol\\":\\"' + PROTOCOL + '\\"'
    if source.count(capability) != 1:
        raise ValueError("v1 sampling capability anchor must occur exactly once")
    source = source.replace(
        capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V2 + '\\",\\"sampling_v1_protocol\\":\\"' + PROTOCOL + '\\"',
        1,
    )
    counter_anchor = "sampling_mask_bytes = 0, sampling_ns = 0;"
    if source.count(counter_anchor) != 1:
        raise ValueError("v2 counter declaration anchor must occur exactly once")
    source = source.replace(
        counter_anchor,
        "sampling_mask_bytes = 0, sampling_ns = 0, sampling_v2_calls = 0, "
        "sampling_v2_logit_bytes = 0, sampling_v2_mask_bytes = 0, sampling_v2_ns = 0;",
        1,
    )
    command_anchor = '} else if (command == "generate") {'
    if source.count(command_anchor) != 1:
        raise ValueError("v2 generate command anchor must occur exactly once")
    command = r'''        } else if (command == "sample_mask_v2") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0;
            uint64_t draw53 = 0;
            std::string top_k_text, mask_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53 >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("sample_mask_v2 requires consumed temperature min_p top_k top_p draw53 and quoted mask path"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_v2 top_k must be an exact nonnegative integer"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v2 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature < 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 ||
                (temperature == 0.0 ? draw53 != 0 : draw53 >= (uint64_t(1) << 53))) {
                error_reply("sample_mask_v2 parameters are outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v2 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v2 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v2 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v2 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v2 mask read failed"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t selection_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += selection_elapsed_ns;
            sampling_v2_ns += selection_elapsed_ns;
            ++sampling_calls;
            ++sampling_v2_calls;
            sampling_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_v2_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_mask_bytes += selection.mask_bytes;
            sampling_v2_mask_bytes += selection.mask_bytes;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                      << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                      << "\"protocol\":\"full-vocab-min-p-u53-v2\",\"token\":" << selection.token
                      << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                      << ",\"mask_bytes\":" << selection.mask_bytes
                      << ",\"legal_count\":" << selection.legal_count
                      << ",\"kept_count\":" << selection.kept_count
                      << ",\"top_k\":" << top_k_arg << ",\"top_p\":" << top_p
                      << ",\"draw53\":" << draw53 << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
        } else if (command == "generate") {'''
    source = source.replace(command_anchor, command, 1)
    stats_anchor = r'<< ",\"sampling_ns\":" << sampling_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v2 stats output anchor must occur exactly once")
    stats = stats_anchor + \
        r' << ",\"sampling_v2_calls\":" << sampling_v2_calls' + \
        r' << ",\"sampling_v2_logit_bytes\":" << sampling_v2_logit_bytes' + \
        r' << ",\"sampling_v2_mask_bytes\":" << sampling_v2_mask_bytes' + \
        r' << ",\"sampling_v2_ns\":" << sampling_v2_ns'
    source = source.replace(stats_anchor, stats, 1)
    required = (
        ('command == "sample_mask_v2"', 1), (PROTOCOL_V2, 2),
        ('\\"top_k\\":" << top_k_arg', 1), ('\\"top_p\\":" << top_p', 1),
        ('\\"sampling_v2_calls\\":" << sampling_v2_calls', 1),
    )
    if any(source.count(anchor) != count for anchor, count in required):
        raise ValueError("v2 command/capability/receipt anchors are not unique")
    return source


def instrument_min_p_selector_v3(source: str) -> str:
    """Add a DRY-aware v3 command while retaining the v1 and v2 commands."""
    source = instrument_min_p_selector_v2(source)
    capability = 'sampling_protocol\\":\\"' + PROTOCOL_V2 + '\\",\\"sampling_v1_protocol\\":\\"' + PROTOCOL + '\\"'
    if source.count(capability) != 1:
        raise ValueError("v2 capability anchor must occur exactly once")
    source = source.replace(capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V3 + '\\",\\"sampling_v2_protocol\\":\\"' + PROTOCOL_V2 +
        '\\",\\"sampling_v1_protocol\\":\\"' + PROTOCOL + '\\"', 1)
    include_anchor = '#include "min_p_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("v3 selector include anchor must occur exactly once")
    source = source.replace(include_anchor,
        include_anchor + "\n#include <fstream>\n#include <iterator>", 1)
    declaration_anchor = "sampling_v2_mask_bytes = 0, sampling_v2_ns = 0;"
    if source.count(declaration_anchor) != 1:
        raise ValueError("v3 counter declaration anchor must occur exactly once")
    source = source.replace(declaration_anchor,
        "sampling_v2_mask_bytes = 0, sampling_v2_ns = 0, sampling_v3_calls = 0, "
        "sampling_v3_logit_bytes = 0, sampling_v3_mask_bytes = 0, sampling_v3_ns = 0, "
        "dry_penalized_candidates = 0;", 1)
    command_anchor = '} else if (command == "generate") {'
    if source.count(command_anchor) != 1:
        raise ValueError("v3 generate command anchor must occur exactly once")
    command = r'''        } else if (command == "sample_mask_v3") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)) || (in >> extra)) {
                error_reply("sample_mask_v3 requires consumed temperature min_p top_k top_p draw53 mask_path dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_v3 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v3 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature < 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                (temperature == 0.0 ? draw53 != 0 : draw53 >= (uint64_t(1) << 53))) {
                error_reply("sample_mask_v3 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v3 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v3 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v3 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v3 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v3 mask read failed"); continue; }

            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            const size_t expected_history = dry_last_n_arg == 0 ? 0 :
                std::min<uint64_t>(actual_consumed, dry_last_n_arg);
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_v3 history/breaker file is missing or malformed"); continue;
            }
            if ((dry_last_n_arg == 0 && !breakers.empty()) || breakers.size() > history.size()) {
                error_reply("sample_mask_v3 history/breaker count mismatch"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_v3 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select_with_dry(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)});
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v3_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v3_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes;
            sampling_v3_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes;
            sampling_v3_mask_bytes += selection.mask_bytes;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-dry-min-p-u53-v3\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
        } else if (command == "generate") {'''
    source = source.replace(command_anchor, command, 1)
    stats_anchor = r'<< ",\"sampling_v2_ns\":" << sampling_v2_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v3 stats anchor must occur exactly once")
    source = source.replace(stats_anchor, stats_anchor +
        r' << ",\"sampling_v3_calls\":" << sampling_v3_calls' +
        r' << ",\"sampling_v3_logit_bytes\":" << sampling_v3_logit_bytes' +
        r' << ",\"sampling_v3_mask_bytes\":" << sampling_v3_mask_bytes' +
        r' << ",\"sampling_v3_ns\":" << sampling_v3_ns' +
        r' << ",\"dry_penalized_candidates\":" << dry_penalized_candidates', 1)
    if source.count('command == "sample_mask_v3"') != 1 or source.count(PROTOCOL_V3) != 2:
        raise ValueError("v3 command/capability anchors are not unique")
    return source


def instrument_min_p_selector_v4(source: str) -> str:
    """Add an ordered DRY/Min-P/K/Typical/Dyna/Top-P command; retain v1-v3."""
    source = instrument_min_p_selector_v3(source)
    capability = ('sampling_protocol\\":\\"' + PROTOCOL_V3 +
                  '\\",\\"sampling_v2_protocol\\":\\"' + PROTOCOL_V2)
    if source.count(capability) != 1:
        raise ValueError("v3 capability anchor must occur exactly once")
    source = source.replace(capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V4 +
        '\\",\\"sampling_v3_protocol\\":\\"' + PROTOCOL_V3 +
        '\\",\\"sampling_v2_protocol\\":\\"' + PROTOCOL_V2, 1)
    include_anchor = '#include "min_p_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("v4 selector include anchor must occur exactly once")
    # The helper includes are available only in v4 build directories; legacy
    # build modes keep the old header-only dependency closure.
    declaration_anchor = "dry_penalized_candidates = 0;"
    if source.count(declaration_anchor) != 1:
        raise ValueError("v4 counter declaration anchor must occur exactly once")
    source = source.replace(declaration_anchor,
        declaration_anchor + " uint64_t sampling_v4_calls = 0, sampling_v4_logit_bytes = 0, "
        "sampling_v4_mask_bytes = 0, sampling_v4_ns = 0, typical_filter_calls = 0, "
        "dynatemp_filter_calls = 0, dry_penalized_candidates_v4 = 0;", 1)
    command_anchor = '} else if (command == "generate") {'
    if source.count(command_anchor) != 1:
        raise ValueError("v4 generate command anchor must occur exactly once")
    command = r'''        } else if (command == "sample_mask_v4") {
            int64_t expected_consumed = 0;
            uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 1.0, dry_multiplier = 0.0, dry_base = 1.0;
            double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)
                    >> typical_p >> dyna_range >> dyna_exponent) || (in >> extra)) {
                error_reply("sample_mask_v4 requires consumed T min_p top_k top_p draw53 mask_path dryMult dryBase dryAllowed dryN history breakers typicalP dynaRange dynaExponent"); continue;
            }
            const auto top_k_parse = std::from_chars(top_k_text.data(), top_k_text.data() + top_k_text.size(), top_k_arg);
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (top_k_text.empty() || top_k_text[0] == '-' || top_k_parse.ec != std::errc() ||
                top_k_parse.ptr != top_k_text.data() + top_k_text.size() ||
                dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                top_k_arg > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_v4 integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_v4 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range)) {
                error_reply("sample_mask_v4 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_v4 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_v4 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_v4 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_v4 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_v4 mask read failed"); continue; }

            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;
            std::vector<uint32_t> history, breakers;
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max()) || breakers.size() > history.size()) {
                error_reply("sample_mask_v4 history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_v4 history is not the native session tail"); continue; }
            min_p_select::Result selection{};
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                selection = min_p_select::select_v4(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent);
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - selection_start).count();
            sampling_ns += elapsed_ns; sampling_v4_ns += elapsed_ns;
            ++sampling_calls; ++sampling_v4_calls;
            const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
            sampling_logit_bytes += logit_bytes; sampling_v4_logit_bytes += logit_bytes;
            sampling_mask_bytes += selection.mask_bytes; sampling_v4_mask_bytes += selection.mask_bytes;
            if (selection.typical_computed) ++typical_filter_calls;
            if (selection.dynatemp_computed) ++dynatemp_filter_calls;
            dry_penalized_candidates += selection.dry_penalized_candidates;
            dry_penalized_candidates_v4 += selection.dry_penalized_candidates;
            const auto old_precision = std::cout.precision();
            std::cout << std::setprecision(17)
                << "{\"status\":\"ok\",\"operator\":\"full-vocab-min-p\","
                << "\"protocol\":\"full-vocab-typical-dyna-min-p-u53-v4\",\"token\":" << selection.token
                << ",\"consumed\":" << actual_consumed << ",\"logits_count\":" << vocab
                << ",\"mask_bytes\":" << selection.mask_bytes << ",\"legal_count\":" << selection.legal_count
                << ",\"min_p_kept_count\":" << selection.min_p_kept_count
                << ",\"top_k_kept_count\":" << selection.top_k_kept_count
                << ",\"typical_kept_count\":" << selection.typical_kept_count
                << ",\"kept_count\":" << selection.kept_count << ",\"top_k\":" << top_k_arg
                << ",\"top_p\":" << top_p << ",\"draw53\":" << draw53
                << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                << ",\"dry_allowed_length\":" << dry_allowed_arg
                << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                << ",\"dry_penalized_candidates\":" << selection.dry_penalized_candidates
                << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                << ",\"dynatemp_exponent\":" << dyna_exponent
                << ",\"effective_temperature\":" << selection.effective_temperature << "}";
            std::cout.precision(old_precision);
            std::cout << std::endl;
        } else if (command == "generate") {'''
    source = source.replace(command_anchor, command, 1)
    stats_anchor = r'<< ",\"sampling_v3_ns\":" << sampling_v3_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v4 stats anchor must occur exactly once")
    source = source.replace(stats_anchor, stats_anchor +
        r' << ",\"sampling_v4_calls\":" << sampling_v4_calls' +
        r' << ",\"sampling_v4_logit_bytes\":" << sampling_v4_logit_bytes' +
        r' << ",\"sampling_v4_mask_bytes\":" << sampling_v4_mask_bytes' +
        r' << ",\"sampling_v4_ns\":" << sampling_v4_ns' +
        r' << ",\"typical_filter_calls\":" << typical_filter_calls' +
        r' << ",\"dynatemp_filter_calls\":" << dynatemp_filter_calls' +
        r' << ",\"dry_penalized_candidates_v4\":" << dry_penalized_candidates_v4', 1)
    if source.count('command == "sample_mask_v4"') != 1 or source.count(PROTOCOL_V4) != 2:
        raise ValueError("v4 command/capability anchors are not unique")
    return source


def instrument_min_p_selector_v5(source: str) -> str:
    """Add the versioned post-Top-P XTC command while preserving v1-v4."""
    source = instrument_min_p_selector_v4(source)
    capability = ('sampling_protocol\\":\\"' + PROTOCOL_V4 +
                  '\\",\\"sampling_v3_protocol\\":\\"' + PROTOCOL_V3)
    if source.count(capability) != 1:
        raise ValueError("v4 capability anchor must occur exactly once for v5")
    source = source.replace(capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V5 +
        '\\",\\"sampling_v4_protocol\\":\\"' + PROTOCOL_V4 +
        '\\",\\"sampling_v3_protocol\\":\\"' + PROTOCOL_V3, 1)

    counter_anchor = "dry_penalized_candidates_v4 = 0;"
    if source.count(counter_anchor) != 1:
        raise ValueError("v5 counter declaration anchor must occur exactly once")
    source = source.replace(counter_anchor, counter_anchor +
        " uint64_t sampling_v5_calls = 0, sampling_v5_logit_bytes = 0, "
        "sampling_v5_mask_bytes = 0, sampling_v5_ns = 0, "
        "sampling_v5_activation_draws = 0, sampling_v5_activated_calls = 0, "
        "sampling_v5_removed_candidates = 0, typical_filter_calls_v5 = 0, "
        "dynatemp_filter_calls_v5 = 0, dry_penalized_candidates_v5 = 0;", 1)

    generate_anchor = '} else if (command == "generate") {'
    command_start = '} else if (command == "sample_mask_v4") {'
    if source.count(generate_anchor) != 1 or source.count(command_start) != 1:
        raise ValueError("v5 command insertion anchors are not unique")
    start = source.index(command_start)
    end = source.index(generate_anchor, start)
    v5_command = source[start:end]
    v5_command = v5_command.replace("sample_mask_v4", "sample_mask_v5")
    v5_command = v5_command.replace(PROTOCOL_V4, PROTOCOL_V5)
    v5_command = v5_command.replace(
        "uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0;",
        "uint64_t top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0, draw53 = 0, "
        "draw53_second = min_p_select::kXtcSecondDrawAbsent;")
    v5_command = v5_command.replace(
        "double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0;",
        "double typical_p = 1.0, dyna_range = 0.0, dyna_exponent = 1.0; "
        "double xtc_probability = 0.0, xtc_threshold = 0.0;")
    parse_anchor = ">> typical_p >> dyna_range >> dyna_exponent) || (in >> extra)) {"
    parse_replacement = (">> typical_p >> dyna_range >> dyna_exponent >> xtc_probability "
                        ">> xtc_threshold >> draw53_second) || (in >> extra)) {")
    if v5_command.count(parse_anchor) != 1:
        raise ValueError("v5 parse extension anchor must occur exactly once")
    v5_command = v5_command.replace(parse_anchor, parse_replacement, 1)
    v5_command = v5_command.replace(
        "sample_mask_v5 requires consumed T min_p top_k top_p draw53 mask_path dryMult dryBase dryAllowed dryN history breakers typicalP dynaRange dynaExponent",
        "sample_mask_v5 requires v4 arguments followed by xtcProbability xtcThreshold secondDraw")
    draw_check = "draw53 >= (uint64_t(1) << 53) ||"
    if v5_command.count(draw_check) != 1:
        raise ValueError("v5 first-draw validation anchor must occur exactly once")
    v5_command = v5_command.replace(draw_check,
        "draw53 >= (uint64_t(1) << 53) || "
        "(draw53_second >= (uint64_t(1) << 53) && "
        "draw53_second != min_p_select::kXtcSecondDrawAbsent) ||", 1)
    dyna_validation = "!std::isfinite(temperature + dyna_range)) {"
    if v5_command.count(dyna_validation) != 1:
        raise ValueError("v5 XTC numeric validation anchor must occur exactly once")
    v5_command = v5_command.replace(dyna_validation,
        "!std::isfinite(temperature + dyna_range) || "
        "!std::isfinite(xtc_probability) || xtc_probability < 0.0 || xtc_probability > 1.0 || "
        "!std::isfinite(xtc_threshold) || xtc_threshold < 0.0 || xtc_threshold > 1.0) {", 1)
    v5_command = v5_command.replace("min_p_select::Result selection{};", "min_p_select::V5Result selection{};", 1)
    select_call = "selection = min_p_select::select_v4(frontier.data(), vocab, mask.data(), mask.size(),\n"
    if v5_command.count(select_call) != 1:
        raise ValueError("v5 selector call anchor must occur exactly once")
    v5_command = v5_command.replace(select_call,
        "selection = min_p_select::select_v5(frontier.data(), vocab, mask.data(), mask.size(),\n", 1)
    draw_call = ("temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,\n"
                 "                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,\n")
    if v5_command.count(draw_call) != 1:
        raise ValueError("v5 selector draw argument anchor must occur exactly once")
    v5_command = v5_command.replace(draw_call,
        "temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53, draw53_second,\n"
        "                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,\n", 1)
    dyna_call = "typical_p, dyna_range, dyna_exponent);"
    if v5_command.count(dyna_call) != 1:
        raise ValueError("v5 selector XTC argument anchor must occur exactly once")
    v5_command = v5_command.replace(dyna_call,
        "typical_p, dyna_range, dyna_exponent, xtc_probability, xtc_threshold);", 1)
    v5_command = v5_command.replace("selection.", "selection.base.")
    v5_command = v5_command.replace("selection.base.base.", "selection.base.")
    v5_command = v5_command.replace("selection.base.dry_penalized_candidates", "selection.base.dry_penalized_candidates")
    v5_command = v5_command.replace("sampling_v4_ns += elapsed_ns", "sampling_v5_ns += elapsed_ns")
    v5_command = v5_command.replace("++sampling_v4_calls", "++sampling_v5_calls")
    v5_command = v5_command.replace("sampling_v4_logit_bytes += logit_bytes", "sampling_v5_logit_bytes += logit_bytes")
    v5_command = v5_command.replace("sampling_v4_mask_bytes += selection.base.mask_bytes", "sampling_v5_mask_bytes += selection.base.mask_bytes")
    v5_command = v5_command.replace("++typical_filter_calls", "++typical_filter_calls_v5")
    v5_command = v5_command.replace("++dynatemp_filter_calls", "++dynatemp_filter_calls_v5")
    v5_command = v5_command.replace("dry_penalized_candidates_v4 += selection.base.dry_penalized_candidates",
        "dry_penalized_candidates_v5 += selection.base.dry_penalized_candidates")
    v5_command = v5_command.replace("\"protocol\":\"" + PROTOCOL_V5 + "\",\"token\":\"",
        "\"protocol\":\"" + PROTOCOL_V5 + "\",\"token\":\"")
    # The response body uses the v4 fields from `base` plus explicit v5 draw/XTC receipts.
    v5_command = _append_v5_response_receipt(v5_command)
    v5_command = v5_command.replace("sampling_ns += elapsed_ns; sampling_v5_ns += elapsed_ns;",
                                    "sampling_ns += elapsed_ns; sampling_v5_ns += elapsed_ns;")
    counter_after = "dry_penalized_candidates += selection.base.dry_penalized_candidates;\n            dry_penalized_candidates_v5 += selection.base.dry_penalized_candidates;"
    if counter_after not in v5_command:
        raise ValueError("v5 dry counter anchor must occur exactly once")
    v5_command = v5_command.replace(counter_after, counter_after +
        "\n            sampling_v5_activation_draws += selection.activation_draws;"
        "\n            sampling_v5_activated_calls += selection.activated ? 1 : 0;"
        "\n            sampling_v5_removed_candidates += selection.removed_count;", 1)
    source = source.replace(generate_anchor, v5_command + generate_anchor, 1)

    stats_anchor = r'<< ",\"sampling_v4_ns\":" << sampling_v4_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v5 statistics anchor must occur exactly once")
    source = source.replace(stats_anchor, stats_anchor +
        r' << ",\"sampling_v5_calls\":" << sampling_v5_calls' +
        r' << ",\"sampling_v5_logit_bytes\":" << sampling_v5_logit_bytes' +
        r' << ",\"sampling_v5_mask_bytes\":" << sampling_v5_mask_bytes' +
        r' << ",\"sampling_v5_ns\":" << sampling_v5_ns' +
        r' << ",\"sampling_v5_activation_draws\":" << sampling_v5_activation_draws' +
        r' << ",\"sampling_v5_activated_calls\":" << sampling_v5_activated_calls' +
        r' << ",\"sampling_v5_removed_candidates\":" << sampling_v5_removed_candidates' +
        r' << ",\"typical_filter_calls_v5\":" << typical_filter_calls_v5' +
        r' << ",\"dynatemp_filter_calls_v5\":" << dynatemp_filter_calls_v5' +
        r' << ",\"dry_penalized_candidates_v5\":" << dry_penalized_candidates_v5', 1)
    if source.count('command == "sample_mask_v5"') != 1 or source.count(PROTOCOL_V5) != 2:
        raise ValueError("v5 command/capability anchors are not unique")
    return source


def instrument_min_p_selector_v6(source: str) -> str:
    """Add standalone Mirostat2 selection while retaining v1-v5 commands."""
    source = instrument_min_p_selector_v5(source)
    capability = ('sampling_protocol\\":\\"' + PROTOCOL_V5 +
                  '\\",\\"sampling_v4_protocol\\":\\"' + PROTOCOL_V4)
    if source.count(capability) != 1:
        raise ValueError("v5 capability anchor must occur exactly once for v6")
    source = source.replace(capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V6 +
        '\\",\\"mirostat2_protocol\\":\\"' + PROTOCOL_V6 +
        '\\",\\"sampling_v5_protocol\\":\\"' + PROTOCOL_V5 +
        '\\",\\"sampling_v4_protocol\\":\\"' + PROTOCOL_V4, 1)
    ready_close = '\\",\\"sampling_v1_protocol\\":\\"' + PROTOCOL + '\\"}'
    if source.count(ready_close) != 1:
        raise ValueError("v5 ready capability terminator must occur exactly once")
    source = source.replace(ready_close,
        '\\",\\"sampling_v1_protocol\\":\\"' + PROTOCOL +
        '\\",\\"mirostat2_feedback_protocol\\":\\"mirostat2-state-v1' +
        '\\",\\"mirostat2_feedback_version\\":1}', 1)

    include_anchor = '#include "min_p_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("Mirostat2 include anchor must occur exactly once")
    source = source.replace(include_anchor,
        include_anchor + '\n#include "mirostat2_select.h"', 1)

    counter_anchor = "dry_penalized_candidates_v5 = 0;"
    if source.count(counter_anchor) != 1:
        raise ValueError("v6 counter insertion anchor must occur exactly once")
    source = source.replace(counter_anchor, counter_anchor +
        " uint64_t sampling_mirostat2_calls = 0, sampling_mirostat2_logit_bytes = 0, "
        "sampling_mirostat2_mask_bytes = 0, sampling_mirostat2_ns = 0;", 1)

    generate_anchor = '} else if (command == "generate") {'
    if source.count(generate_anchor) != 1:
        raise ValueError("v6 generate insertion anchor must occur exactly once")
    command = r'''        } else if (command == "sample_mask_mirostat2") {
            int64_t expected_consumed = 0;
            uint64_t draw53 = 0, dry_allowed_arg = 0, dry_last_n_arg = 0;
            double temperature = 0.0, tau = 0.0, eta = 0.0, mu = 0.0;
            double dry_multiplier = 0.0, dry_base = 1.0;
            std::string dry_allowed_text, dry_last_n_text, mask_path, history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> draw53 >> std::quoted(mask_path)
                    >> tau >> eta >> mu >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path)) || (in >> extra)) {
                error_reply("sample_mask_mirostat2 requires consumed T draw53 mask_path tau eta mu dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path"); continue;
            }
            const auto allowed_parse = std::from_chars(dry_allowed_text.data(), dry_allowed_text.data() + dry_allowed_text.size(), dry_allowed_arg);
            const auto last_n_parse = std::from_chars(dry_last_n_text.data(), dry_last_n_text.data() + dry_last_n_text.size(), dry_last_n_arg);
            if (dry_allowed_text.empty() || dry_allowed_text[0] == '-' || allowed_parse.ec != std::errc() ||
                allowed_parse.ptr != dry_allowed_text.data() + dry_allowed_text.size() ||
                dry_last_n_text.empty() || dry_last_n_text[0] == '-' || last_n_parse.ec != std::errc() ||
                last_n_parse.ptr != dry_last_n_text.data() + dry_last_n_text.size() ||
                dry_allowed_arg > UINT32_MAX || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_mirostat2 DRY integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_mirostat2 consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(tau) || tau <= 0.0 || !std::isfinite(eta) || eta < 0.0 ||
                !std::isfinite(mu) || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0) {
                error_reply("sample_mask_mirostat2 parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_mirostat2 frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_mirostat2 could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_mirostat2 truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_mirostat2 extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_mirostat2 mask read failed"); continue; }
            const size_t used_bits = vocab % 8;
            if (used_bits != 0 && (mask.back() & static_cast<uint8_t>(~((1u << used_bits) - 1u))) != 0) {
                error_reply("sample_mask_mirostat2 mask has nonzero padding bits"); continue;
            }

            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                }
                return true;
            };
            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_mirostat2 history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_mirostat2 history is not the native session tail"); continue; }

            min_p_select::DryConfig dry{dry_multiplier, dry_base,
                static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)};
            size_t dry_penalized_candidates = 0;
            std::vector<float> adjusted;
            std::vector<uint32_t> legal_ids;
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab, history, breakers,
                    dry, &dry_penalized_candidates);
                legal_ids.reserve(vocab);
                for (size_t token = 0; token < vocab; ++token) {
                    if (!min_p_select::finite_binary32(frontier[token]))
                        throw std::invalid_argument("Mirostat2 logits contain a nonfinite score");
                    if (min_p_select::legal(mask.data(), token)) legal_ids.push_back(static_cast<uint32_t>(token));
                }
                const auto selection = mirostat2_select::choose(adjusted, legal_ids,
                    temperature, draw53, mu, tau, eta);
                const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - selection_start).count();
                sampling_ns += elapsed_ns; sampling_mirostat2_ns += elapsed_ns;
                ++sampling_calls; ++sampling_mirostat2_calls;
                const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
                sampling_logit_bytes += logit_bytes; sampling_mirostat2_logit_bytes += logit_bytes;
                sampling_mask_bytes += expected_mask_bytes; sampling_mirostat2_mask_bytes += expected_mask_bytes;
                const auto old_precision = std::cout.precision();
                std::cout << std::setprecision(17)
                    << "{\"status\":\"ok\",\"operator\":\"full-vocab-mirostat2\","
                    << "\"protocol\":\"full-vocab-mirostat2-u53-v6\","
                    << "\"feedback_protocol\":\"mirostat2-state-v1\",\"feedback_version\":1"
                    << ",\"token\":" << selection.token << ",\"consumed\":" << actual_consumed
                    << ",\"logits_count\":" << vocab << ",\"mask_bytes\":" << expected_mask_bytes
                    << ",\"legal_count\":" << legal_ids.size() << ",\"kept_count\":" << selection.kept_ids.size()
                    << ",\"legal_total_weight\":" << selection.legal_total_weight
                    << ",\"total_weight\":" << selection.total_weight << ",\"draw53\":" << draw53
                    << ",\"draw_count\":1,\"temperature\":" << temperature << ",\"tau\":" << tau
                    << ",\"eta\":" << eta << ",\"mu\":" << mu << ",\"next_mu\":" << selection.next_mu
                    << ",\"selected_log_probability\":" << selection.selected_log_probability
                    << ",\"selected_probability\":" << selection.selected_probability
                    << ",\"surprise_bits\":" << selection.surprise_bits
                    << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                    << ",\"dry_allowed_length\":" << dry_allowed_arg
                    << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                    << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                    << ",\"dry_penalized_candidates\":" << dry_penalized_candidates << "}";
                std::cout.precision(old_precision);
                std::cout << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
        } else if (command == "generate") {'''
    if command.count('command == "sample_mask_mirostat2"') != 1:
        raise ValueError("v6 command text does not contain its command exactly once")
    source = source.replace(generate_anchor, command, 1)

    stats_anchor = r'<< ",\"sampling_v5_ns\":" << sampling_v5_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v6 stats insertion anchor must occur exactly once")
    source = source.replace(stats_anchor, stats_anchor +
        r' << ",\"sampling_mirostat2_calls\":" << sampling_mirostat2_calls' +
        r' << ",\"sampling_mirostat2_logit_bytes\":" << sampling_mirostat2_logit_bytes' +
        r' << ",\"sampling_mirostat2_mask_bytes\":" << sampling_mirostat2_mask_bytes' +
        r' << ",\"sampling_mirostat2_ns\":" << sampling_mirostat2_ns', 1)
    # One capability string is exposed as the general sampling protocol, a
    # second as the Mirostat command capability, and a third in each success
    # receipt's protocol field.
    if source.count('command == "sample_mask_mirostat2"') != 1 or source.count(PROTOCOL_V6) != 3:
        raise ValueError("v6 command/capability anchors are not unique")
    return source


def instrument_min_p_selector_v7(source: str) -> str:
    """Add Adaptive-P over the exact post-Top-P v4 prepared weights."""
    source = instrument_min_p_selector_v6(source)
    capability = ('sampling_protocol\\":\\"' + PROTOCOL_V6 +
                  '\\",\\"mirostat2_protocol\\":\\"' + PROTOCOL_V6)
    if source.count(capability) != 1:
        raise ValueError("v6 ready capability anchor must occur exactly once for v7")
    source = source.replace(capability,
        'sampling_protocol\\":\\"' + PROTOCOL_V7 +
        '\\",\\"adaptive_p_protocol\\":\\"' + PROTOCOL_V7 +
        '\\",\\"mirostat2_protocol\\":\\"' + PROTOCOL_V6 +
        '\\",\\"sampling_v6_protocol\\":\\"' + PROTOCOL_V6, 1)
    source = insert_adaptive_p_ready_fields(source)

    include_anchor = '#include "mirostat2_select.h"'
    if source.count(include_anchor) != 1:
        raise ValueError("v6 helper include anchor must occur exactly once for v7")
    source = source.replace(include_anchor,
        include_anchor + '\n#include "adaptive_p_select.h"', 1)

    counter_anchor = 'uint64_t sampling_mirostat2_calls = 0, sampling_mirostat2_logit_bytes = 0, '
    if source.count(counter_anchor) != 1:
        raise ValueError("v6 counter declaration anchor must occur exactly once for v7")
    source = source.replace(counter_anchor,
        'uint64_t sampling_adaptive_p_calls = 0, sampling_adaptive_p_logit_bytes = 0, '
        'sampling_adaptive_p_mask_bytes = 0, sampling_adaptive_p_ns = 0; ' + counter_anchor, 1)

    generate_anchor = '} else if (command == "generate") {'
    if source.count(generate_anchor) != 1:
        raise ValueError("v7 generate insertion anchor must occur exactly once")
    command = r'''        } else if (command == "sample_mask_adaptive_p") {
            int64_t expected_consumed = 0;
            uint64_t draw53 = 0, top_k_arg = 0, dry_allowed_arg = 0, dry_last_n_arg = 0;
            double temperature = 0.0, min_p = 0.0, top_p = 0.0;
            double dry_multiplier = 0.0, dry_base = 1.0, typical_p = 1.0;
            double dyna_range = 0.0, dyna_exponent = 1.0, adaptive_target = 0.0;
            double adaptive_decay = 0.0, state_weighted_sum = 0.0, state_total_weight = 0.0;
            std::string top_k_text, dry_allowed_text, dry_last_n_text, mask_path;
            std::string history_path, breakers_path, extra;
            if (!(in >> expected_consumed >> temperature >> min_p >> top_k_text >> top_p >> draw53
                    >> std::quoted(mask_path) >> dry_multiplier >> dry_base >> dry_allowed_text >> dry_last_n_text
                    >> std::quoted(history_path) >> std::quoted(breakers_path) >> typical_p >> dyna_range
                    >> dyna_exponent >> adaptive_target >> adaptive_decay >> state_weighted_sum >> state_total_weight)
                    || (in >> extra)) {
                error_reply("sample_mask_adaptive_p requires consumed T min_p top_k top_p draw53 mask_path dry_multiplier dry_base dry_allowed dry_last_n history_path breakers_path typical_p dyna_range dyna_exponent target decay weighted_sum total_weight"); continue;
            }
            const auto parse_u64 = [](const std::string& text, uint64_t& value) {
                if (text.empty() || text[0] == '-') return false;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
                return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
            };
            if (!parse_u64(top_k_text, top_k_arg) || top_k_arg > std::numeric_limits<size_t>::max() ||
                !parse_u64(dry_allowed_text, dry_allowed_arg) || dry_allowed_arg > UINT32_MAX ||
                !parse_u64(dry_last_n_text, dry_last_n_arg) || dry_last_n_arg > UINT32_MAX) {
                error_reply("sample_mask_adaptive_p integer parameter is out of range"); continue;
            }
            const uint64_t actual_consumed = session.ids().size();
            if (expected_consumed <= 0 || static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("sample_mask_adaptive_p consumed count mismatch"); continue;
            }
            if (!std::isfinite(temperature) || temperature <= 0.0 ||
                !std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0 ||
                !std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0 || draw53 >= (uint64_t(1) << 53) ||
                !std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
                !std::isfinite(dry_base) || dry_base < 1.0 ||
                !std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0 ||
                !std::isfinite(dyna_range) || dyna_range < 0.0 ||
                !std::isfinite(dyna_exponent) || dyna_exponent <= 0.0 ||
                !std::isfinite(temperature + dyna_range) ||
                !std::isfinite(adaptive_target) || adaptive_target > 1.0 ||
                !std::isfinite(adaptive_decay) || adaptive_decay < 0.0 || adaptive_decay > 0.99 ||
                !std::isfinite(state_weighted_sum) || !std::isfinite(state_total_weight) || state_total_weight <= 0.0) {
                error_reply("sample_mask_adaptive_p parameter is outside the protocol range"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("sample_mask_adaptive_p frontier extent mismatch"); continue;
            }
            const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("sample_mask_adaptive_p could not open mask file"); continue; }
            std::vector<uint8_t> mask(expected_mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(expected_mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(expected_mask_bytes)) {
                error_reply("sample_mask_adaptive_p truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("sample_mask_adaptive_p extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("sample_mask_adaptive_p mask read failed"); continue; }
            const size_t used_bits = vocab % 8;
            if (used_bits != 0 && (mask.back() & static_cast<uint8_t>(~((1u << used_bits) - 1u))) != 0) {
                error_reply("sample_mask_adaptive_p mask has nonzero padding bits"); continue;
            }
            const bool dry_active = dry_multiplier > 0.0 && dry_last_n_arg > 0;
            const size_t expected_history = adaptive_p_select::expected_dry_history_count(
                actual_consumed, dry_multiplier, dry_last_n_arg);
            const auto read_u32_file = [](const std::string& path, std::vector<uint32_t>& values,
                                          size_t max_count, size_t exact_count) -> bool {
                std::ifstream file(path, std::ios::binary);
                if (!file) return false;
                file.seekg(0, std::ios::end);
                const std::streamoff end = file.tellg();
                if (end < 0 || end % 4 != 0 || static_cast<uint64_t>(end / 4) > max_count ||
                    (exact_count != std::numeric_limits<size_t>::max() &&
                     static_cast<uint64_t>(end / 4) != exact_count)) return false;
                file.seekg(0, std::ios::beg);
                std::vector<uint8_t> bytes(static_cast<size_t>(end));
                if (end != 0) file.read(reinterpret_cast<char*>(bytes.data()), end);
                if (file.bad() || file.gcount() != end) return false;
                values.resize(bytes.size() / 4);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<uint32_t>(bytes[4*i]) |
                        (static_cast<uint32_t>(bytes[4*i+1]) << 8) |
                        (static_cast<uint32_t>(bytes[4*i+2]) << 16) |
                        (static_cast<uint32_t>(bytes[4*i+3]) << 24);
                return true;
            };
            std::vector<uint32_t> history, breakers;
            if (!read_u32_file(history_path, history, expected_history, expected_history) ||
                !read_u32_file(breakers_path, breakers, expected_history,
                    std::numeric_limits<size_t>::max())) {
                error_reply("sample_mask_adaptive_p history/breaker file is missing or malformed"); continue;
            }
            const auto& native_ids = session.ids();
            bool history_matches = true;
            for (size_t i = 0; i < history.size(); ++i) {
                if (history[i] != static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])) {
                    history_matches = false; break;
                }
            }
            if (!history_matches) { error_reply("sample_mask_adaptive_p history is not the native session tail"); continue; }
            if (dry_active) {
                uint32_t prior = 0; bool first_breaker = true;
                for (uint32_t breaker : breakers) {
                    if (breaker >= vocab || (!first_breaker && breaker <= prior)) { history_matches = false; break; }
                    first_breaker = false; prior = breaker;
                }
                if (!history_matches) { error_reply("sample_mask_adaptive_p breakers are not sorted unique vocabulary IDs"); continue; }
            }

            min_p_select::DryConfig dry{dry_multiplier, dry_base,
                static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)};
            adaptive_p_select::State state{state_weighted_sum, state_total_weight};
            size_t dry_penalized_candidates = 0;
            const auto selection_start = std::chrono::steady_clock::now();
            try {
                const auto prepared = min_p_select::select_v4(frontier.data(), vocab, mask.data(),
                    expected_mask_bytes, temperature, min_p, static_cast<size_t>(top_k_arg), top_p,
                    draw53, history, breakers, dry, typical_p, dyna_range, dyna_exponent, true);
                dry_penalized_candidates = prepared.dry_penalized_candidates;
                const auto selection = adaptive_p_select::select_prepared(
                    prepared.v5_post_top_p_weights, prepared.v5_post_top_p_ids, state,
                    adaptive_target, adaptive_decay, draw53);
                const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - selection_start).count();
                sampling_ns += elapsed_ns; sampling_adaptive_p_ns += elapsed_ns;
                ++sampling_calls; ++sampling_adaptive_p_calls;
                const uint64_t logit_bytes = static_cast<uint64_t>(vocab) * sizeof(float);
                sampling_logit_bytes += logit_bytes; sampling_adaptive_p_logit_bytes += logit_bytes;
                sampling_mask_bytes += expected_mask_bytes; sampling_adaptive_p_mask_bytes += expected_mask_bytes;
                const auto old_precision = std::cout.precision();
                std::cout << std::setprecision(17)
                    << "{\"status\":\"ok\",\"operator\":\"full-vocab-adaptive-p\","
                    << "\"protocol\":\"full-vocab-adaptive-p-u53-v7\","
                    << "\"feedback_protocol\":\"adaptive-p-state-v1\",\"feedback_version\":1"
                    << ",\"token\":" << selection.token << ",\"consumed\":" << actual_consumed
                    << ",\"logits_count\":" << vocab << ",\"mask_bytes\":" << expected_mask_bytes
                    << ",\"legal_count\":" << prepared.legal_count
                    << ",\"min_p_kept_count\":" << prepared.min_p_kept_count
                    << ",\"top_k_kept_count\":" << prepared.top_k_kept_count
                    << ",\"typical_kept_count\":" << prepared.typical_kept_count
                    << ",\"kept_count\":" << selection.legal_count
                    << ",\"categorical_total_weight\":" << selection.total_weight
                    << ",\"draw53\":" << draw53 << ",\"draw_count\":1"
                    << ",\"temperature\":" << temperature << ",\"effective_temperature\":" << prepared.effective_temperature
                    << ",\"min_p\":" << min_p << ",\"top_k\":" << top_k_arg << ",\"top_p\":" << top_p
                    << ",\"typical_p\":" << typical_p << ",\"dynatemp_range\":" << dyna_range
                    << ",\"dynatemp_exponent\":" << dyna_exponent
                    << ",\"adaptive_target\":" << adaptive_target << ",\"adaptive_decay\":" << adaptive_decay
                    << ",\"original_probability\":" << selection.original_probability
                    << ",\"adapted_target\":";
                if (selection.adapted_target) std::cout << *selection.adapted_target;
                else std::cout << "null";
                std::cout << ",\"state_weighted_sum\":" << selection.source_weighted_sum
                    << ",\"state_total_weight\":" << selection.source_total_weight
                    << ",\"next_weighted_sum\":" << selection.next_weighted_sum
                    << ",\"next_total_weight\":" << selection.next_total_weight
                    << ",\"dry_multiplier\":" << dry_multiplier << ",\"dry_base\":" << dry_base
                    << ",\"dry_allowed_length\":" << dry_allowed_arg
                    << ",\"dry_penalty_last_n\":" << dry_last_n_arg
                    << ",\"dry_history_count\":" << history.size() << ",\"dry_breaker_count\":" << breakers.size()
                    << ",\"dry_penalized_candidates\":" << dry_penalized_candidates << "}";
                std::cout.precision(old_precision);
                std::cout << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
        } else if (command == "generate") {'''
    source = source.replace(generate_anchor, command, 1)
    stats_anchor = r'<< ",\"sampling_mirostat2_ns\":" << sampling_mirostat2_ns'
    if source.count(stats_anchor) != 1:
        raise ValueError("v6 stats anchor must occur exactly once for v7")
    source = source.replace(stats_anchor, stats_anchor +
        r' << ",\"sampling_adaptive_p_calls\":" << sampling_adaptive_p_calls' +
        r' << ",\"sampling_adaptive_p_logit_bytes\":" << sampling_adaptive_p_logit_bytes' +
        r' << ",\"sampling_adaptive_p_mask_bytes\":" << sampling_adaptive_p_mask_bytes' +
        r' << ",\"sampling_adaptive_p_ns\":" << sampling_adaptive_p_ns', 1)
    if source.count('command == "sample_mask_adaptive_p"') != 1 or source.count(PROTOCOL_V7) != 3:
        raise ValueError("v7 command/capability anchors are not unique")
    return source


def insert_adaptive_p_ready_fields(source: str) -> str:
    """Insert v7 feedback capabilities before the retained Mirostat version."""
    feedback_anchor = '\\",\\"mirostat2_feedback_version\\":1}'
    if source.count(feedback_anchor) != 1:
        raise ValueError("v6 feedback ready terminator must occur exactly once for v7")
    replacement = (
        '\\",\\"adaptive_p_feedback_protocol\\":\\"adaptive-p-state-v1' +
        '\\",\\"adaptive_p_feedback_version\\":1,\\"mirostat2_feedback_version\\":1}'
    )
    return source.replace(feedback_anchor, replacement, 1)


def _row(receipt: dict, label: str) -> dict:
    matches = [row for row in receipt.get("commands", []) if row.get("label") == label]
    if len(matches) != 1 or matches[0].get("exit_code") != 0 or not isinstance(matches[0].get("argv"), list):
        raise ValueError(f"receipt requires one successful {label} command")
    return matches[0]


def _verify_inputs(receipt: dict) -> dict[str, str]:
    items = receipt.get("verified_unchanged_inputs")
    if not isinstance(items, dict) or not items:
        raise ValueError("receipt lacks verified unchanged input hashes")
    verified = {}
    for raw_path, expected in sorted(items.items()):
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"pinned input is missing or changed: {path}")
        verified[str(path)] = expected
    return verified


def build(receipt_path: Path, driver_path: Path, header_path: Path, output_dir: Path,
          protocol: str = "v1", typical_header_path: Path | None = None,
          dynatemp_header_path: Path | None = None,
          xtc_header_path: Path | None = None,
          mirostat2_header_path: Path | None = None,
          adaptive_p_header_path: Path | None = None) -> dict:
    receipt_path = receipt_path.resolve(strict=True)
    driver_path = driver_path.resolve(strict=True)
    header_path = header_path.resolve(strict=True)
    if protocol in ("v4", "v5", "v6", "v7"):
        if typical_header_path is None or dynatemp_header_path is None:
            raise ValueError("v4-v6 requires the frozen Typical and DynaTemp helper headers")
        typical_header_path = typical_header_path.resolve(strict=True)
        dynatemp_header_path = dynatemp_header_path.resolve(strict=True)
    if protocol in ("v5", "v6", "v7"):
        if xtc_header_path is None:
            raise ValueError("v5/v6 requires the frozen XTC helper header")
        xtc_header_path = xtc_header_path.resolve(strict=True)
    if protocol in ("v6", "v7"):
        if mirostat2_header_path is None:
            raise ValueError("v6-v7 requires the frozen Mirostat2 helper header")
        mirostat2_header_path = mirostat2_header_path.resolve(strict=True)
    if protocol == "v7":
        if adaptive_p_header_path is None:
            raise ValueError("v7 requires the frozen Adaptive-P helper header")
        adaptive_p_header_path = adaptive_p_header_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "BUILT_NOT_EXECUTED":
        raise ValueError("source receipt must be frozen BUILT_NOT_EXECUTED")
    verified_inputs = _verify_inputs(receipt)
    pinned_driver = receipt.get("generated_driver", {})
    if pinned_driver.get("path") != str(driver_path) or pinned_driver.get("sha256") != sha256(driver_path):
        raise ValueError("driver does not match receipt's generated selector driver")
    compile_row = _row(receipt, "compile-selector-driver")
    link_row = _row(receipt, "link-selector-sut")
    original_source = driver_path.read_text(encoding="utf-8")
    if protocol == "v1":
        source = instrument_min_p_selector(original_source)
        selected_protocol = PROTOCOL
        binary_name = "strata-batched-branch-lazy-min-p"
        command_label = "compile-min-p-driver"
    elif protocol == "v2":
        source = instrument_min_p_selector_v2(original_source)
        selected_protocol = PROTOCOL_V2
        binary_name = "strata-batched-branch-lazy-min-p-v2"
        command_label = "compile-min-p-top-k-top-p-driver"
    elif protocol == "v3":
        source = instrument_min_p_selector_v3(original_source)
        selected_protocol = PROTOCOL_V3
        binary_name = "strata-batched-branch-lazy-dry-min-p-v3"
        command_label = "compile-dry-min-p-driver"
    elif protocol == "v4":
        source = instrument_min_p_selector_v4(original_source)
        selected_protocol = PROTOCOL_V4
        binary_name = "strata-batched-branch-lazy-typical-dyna-min-p-v4"
        command_label = "compile-typical-dynatemp-driver"
    elif protocol == "v5":
        source = instrument_min_p_selector_v5(original_source)
        selected_protocol = PROTOCOL_V5
        binary_name = "strata-batched-branch-lazy-xtc-min-p-v5"
        command_label = "compile-xtc-min-p-driver"
    elif protocol == "v6":
        source = instrument_min_p_selector_v6(original_source)
        selected_protocol = PROTOCOL_V6
        binary_name = "strata-batched-branch-lazy-mirostat2-v6"
        command_label = "compile-mirostat2-driver"
    elif protocol == "v7":
        source = instrument_min_p_selector_v7(original_source)
        selected_protocol = PROTOCOL_V7
        binary_name = "strata-batched-branch-lazy-adaptive-p-v7"
        command_label = "compile-adaptive-p-driver"
    else:
        raise ValueError("protocol must be one of v1, v2, v3, v4, v5, v6, or v7")

    output_dir.mkdir(parents=True, exist_ok=False)
    generated_cpp = output_dir / "branch_driver.cpp"
    generated_obj = output_dir / "branch_driver.o"
    copied_header = output_dir / "min_p_select.h"
    copied_typical_header = output_dir / "typical_select.h"
    copied_dynatemp_header = output_dir / "dynatemp_select.h"
    copied_xtc_header = output_dir / "xtc_select.h"
    copied_mirostat2_header = output_dir / "mirostat2_select.h"
    copied_adaptive_p_header = output_dir / "adaptive_p_select.h"
    executable = output_dir / binary_name
    generated_cpp.write_text(source, encoding="utf-8", newline="\n")
    copied_header.write_bytes(header_path.read_bytes())
    if protocol in ("v4", "v5", "v6", "v7"):
        copied_typical_header.write_bytes(typical_header_path.read_bytes())
        copied_dynatemp_header.write_bytes(dynatemp_header_path.read_bytes())
    if protocol in ("v5", "v6", "v7"):
        copied_xtc_header.write_bytes(xtc_header_path.read_bytes())
    if protocol == "v6":
        copied_mirostat2_header.write_bytes(mirostat2_header_path.read_bytes())
    if protocol == "v7":
        copied_mirostat2_header.write_bytes(mirostat2_header_path.read_bytes())
        copied_adaptive_p_header.write_bytes(adaptive_p_header_path.read_bytes())

    compile_argv = list(compile_row["argv"])
    original_cpp = str(pinned_driver["path"])
    original_obj = receipt.get("generated_object", {}).get("path")
    if not isinstance(original_obj, str) or compile_argv.count(original_cpp) != 1:
        raise ValueError("recorded compile argv does not identify the selected generated driver")
    compile_argv[compile_argv.index(original_cpp)] = str(generated_cpp)
    obj_outputs = [arg for arg in compile_argv if arg.endswith("/branch_driver.o")]
    if len(obj_outputs) != 1:
        raise ValueError("compile argv must contain exactly one branch_driver.o output")
    compile_argv[compile_argv.index(obj_outputs[0])] = str(generated_obj)
    compile_argv.extend(["-I", str(output_dir)])
    if protocol in ("v4", "v5", "v6", "v7"):
        compile_argv.append("-DMIN_P_SELECT_ENABLE_V4=1")
    if protocol in ("v5", "v6", "v7"):
        compile_argv.append("-DMIN_P_SELECT_ENABLE_V5=1")

    link_argv = list(link_row["argv"])
    if link_argv.count(original_obj) != 1:
        raise ValueError("recorded link argv must use the selected driver object exactly once")
    link_argv[link_argv.index(original_obj)] = str(generated_obj)
    output_indexes = [i for i, item in enumerate(link_argv[:-1]) if item == "-o"]
    if len(output_indexes) != 1:
        raise ValueError("link argv must contain exactly one executable output")
    previous_binary = link_argv[output_indexes[0] + 1]
    link_argv[output_indexes[0] + 1] = str(executable)
    cwd = Path(compile_row.get("cwd", "")).resolve(strict=True)
    if Path(link_row.get("cwd", "")).resolve(strict=True) != cwd:
        raise ValueError("compile and link commands use different cwd values")

    command_records = []
    for label, argv in ((command_label, compile_argv), ("link-min-p-sut", link_argv)):
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
        elapsed = time.monotonic() - start
        stdout_path = output_dir / f"{label}.stdout.log"
        stderr_path = output_dir / f"{label}.stderr.log"
        stdout_path.write_bytes(proc.stdout)
        stderr_path.write_bytes(proc.stderr)
        command_records.append({"label": label, "argv": argv, "cwd": str(cwd),
                                "exit_code": proc.returncode, "seconds": elapsed,
                                "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path)})
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed; inspect {stderr_path}")
    if not executable.is_file() or not generated_obj.is_file():
        raise RuntimeError("compile or link returned zero without creating expected outputs")
    if _verify_inputs(receipt) != verified_inputs:
        raise RuntimeError("a pinned input changed during build")
    additional_headers = {}
    if protocol in ("v4", "v5", "v6", "v7"):
        header_pairs = [(typical_header_path, copied_typical_header),
                        (dynatemp_header_path, copied_dynatemp_header)]
        if protocol in ("v5", "v6", "v7"):
            header_pairs.append((xtc_header_path, copied_xtc_header))
        if protocol in ("v6", "v7"):
            header_pairs.append((mirostat2_header_path, copied_mirostat2_header))
        if protocol == "v7":
            header_pairs.append((adaptive_p_header_path, copied_adaptive_p_header))
        for original, copied in header_pairs:
            digest = sha256(original)
            if sha256(copied) != digest or sha256(original) != digest:
                raise RuntimeError(f"v4 helper header changed during build: {original}")
            additional_headers[str(original)] = {"sha256": digest,
                                                  "copied_path": str(copied)}

    result = {
        "status": "BUILT_NOT_EXECUTED",
        "numerical_qualification": "UNRUN",
        "sampling_protocol": selected_protocol,
        "scope": ("full-vocabulary Min-P v1 command; only the generated driver translation unit is recompiled"
                  if protocol == "v1" else
                  "versioned full-vocabulary Min-P/Top-K/Top-P v2 command; v1 command retained; "
                  "only the generated driver translation unit is recompiled" if protocol == "v2" else
                  "versioned full-vocabulary DRY/Min-P/Top-K/Top-P v3 command; v1 and v2 commands retained; "
                  "only the generated driver translation unit is recompiled" if protocol == "v3" else
                  "versioned DRY/Min-P/Top-K/Typical/DynaTemp/Top-P v4 command; v1-v3 commands retained; "
                  "only the generated driver translation unit is recompiled" if protocol == "v4" else
                "versioned DRY/Min-P/Top-K/Typical/DynaTemp/Top-P/XTC v5 command; v1-v4 retained; "
                "only the generated driver translation unit is recompiled" if protocol == "v5" else
                "versioned standalone DRY/Mirostat2 v6 command; v1-v5 retained; "
                  "only the generated driver translation unit is recompiled" if protocol == "v6" else
                  "versioned DRY/Min-P/Top-K/Typical/DynaTemp/Top-P/Adaptive-P v7 command; v1-v6 retained; "
                  "only the generated driver translation unit is recompiled"),
        "builder_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
        "source_driver": {"path": str(driver_path), "sha256": sha256(driver_path)},
        "header": {"path": str(copied_header), "sha256": sha256(copied_header)},
        "additional_headers": additional_headers,
        "generated_driver": {"path": str(generated_cpp), "sha256": sha256(generated_cpp)},
        "generated_object": {"path": str(generated_obj), "sha256": sha256(generated_obj),
                             "bytes": generated_obj.stat().st_size},
        "verified_unchanged_inputs": verified_inputs,
        "previous_binary": previous_binary,
        "commands": command_records,
        "binary": {"path": str(executable), "sha256": sha256(executable), "bytes": executable.stat().st_size},
    }
    (output_dir / "build.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "min_p_select.h")
    parser.add_argument("--typical-header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "typical_select.h")
    parser.add_argument("--dynatemp-header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "dynatemp_select.h")
    parser.add_argument("--xtc-header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "xtc_select.h")
    parser.add_argument("--mirostat2-header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "mirostat2_select.h")
    parser.add_argument("--adaptive-p-header", type=Path,
                        default=Path(__file__).resolve().parents[1] / "native" / "adaptive_p_select.h")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7"), default="v1")
    args = parser.parse_args(argv)
    try:
        result = build(args.receipt, args.driver, args.header, args.output_dir, args.protocol,
                       args.typical_header, args.dynatemp_header, args.xtc_header, args.mirostat2_header,
                       args.adaptive_p_header)
    except Exception as exc:
        print(f"Min-P native build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
