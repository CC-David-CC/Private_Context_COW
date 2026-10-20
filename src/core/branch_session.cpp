#include "strata/core/branch_session.hpp"
#include "conversation_copy.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/mrope.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>

namespace strata::core {
namespace {
std::mutex snapshot_mutex;
BranchSnapshotCounters snapshot_work;
struct SnapshotLifetime {
    uint64_t payload_bytes = 0;
    SnapshotLifetime() {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        ++snapshot_work.allocations;
        ++snapshot_work.live_images;
    }
    SnapshotLifetime(const SnapshotLifetime&) = delete;
    SnapshotLifetime& operator=(const SnapshotLifetime&) = delete;
    void publish(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        payload_bytes = bytes;
        snapshot_work.payload_capacity_bytes += bytes;
        snapshot_work.peak_payload_capacity_bytes = std::max(snapshot_work.peak_payload_capacity_bytes,
                                                            snapshot_work.payload_capacity_bytes);
    }
    ~SnapshotLifetime() {
        // This member is declared first and destroyed last, after the image's
        // vectors and page owners have actually released their resources.
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        ++snapshot_work.releases;
        --snapshot_work.live_images;
        snapshot_work.payload_capacity_bytes -= payload_bytes;
    }
};
const char* pool_kinds[] = {"k", "v", "k_scale", "v_scale", "indexer_pool"};
bool fail(std::string& error, const char* message) {
    error = std::string("branch session: ") + message;
    return false;
}
bool sync(std::string& error) {
    const auto result = cudaDeviceSynchronize();
    return result == cudaSuccess || fail(error, cudaGetErrorString(result));
}
void* pool_pointer(const QsaState& q, size_t part) {
    if (part == 4) return q.idx_pooled;
    if (part == 2) return q.k_scale;
    if (part == 3) return q.v_scale;
    if (q.kv_hybrid) return part == 0 ? (void*)q.k_q : q.v_q4;
    if (q.kv_q4) return part == 0 ? q.k_q4 : q.v_q4;
    if (q.kv_int8) return part == 0 ? q.k_q : q.v_q;
    return part == 0 ? q.k_pool : q.v_pool;
}
void set_pointer(QsaState& q, size_t part, void* p) {
    if (part == 4) q.idx_pooled = (float*)p;
    else if (part == 2) q.k_scale = (uint16_t*)p;
    else if (part == 3) q.v_scale = (uint16_t*)p;
    else if (q.kv_hybrid) { if (part == 0) q.k_q = (int8_t*)p; else q.v_q4 = (uint8_t*)p; }
    else if (q.kv_q4) { if (part == 0) q.k_q4 = (uint8_t*)p; else q.v_q4 = (uint8_t*)p; }
    else if (q.kv_int8) { if (part == 0) q.k_q = (int8_t*)p; else q.v_q = (int8_t*)p; }
    else { if (part == 0) q.k_pool = (uint16_t*)p; else q.v_pool = (uint16_t*)p; }
}
uint64_t running_bytes(const ConversationCheckpoint& c) {
    return c.gdn.size() + c.ple.size() + c.tails.size() + c.dead.size() + c.block_pos.size();
}
SessionFileIdentity file_identity(const std::string& contract, const std::vector<int64_t>& layout) {
    // Native format detects accidental corruption. Cryptographic file, model
    // and execution fingerprints are checked by the outer API manifest.
    return {session_hash64(contract.data(), contract.size(), 0),
            session_hash64(layout.data(), layout.size()*sizeof(int64_t), 0)};
}
std::vector<ConversationImageKey> conversation_images(const std::vector<VisionPlacement>& inputs) {
    std::vector<ConversationImageKey> result;
    result.reserve(inputs.size());
    for (const auto& image : inputs) {
        // Compact native-format compatibility key only. Full SHA-256 payload,
        // placement and positional bindings remain in the outer manifest.
        const auto digest = vision_input_digest({}, {image});
        result.push_back({image.start, std::stoull(digest.substr(0, 16), nullptr, 16)});
    }
    return result;
}
}

struct BranchSessionSnapshot {
    SnapshotLifetime lifetime;
    std::string contract;
    std::vector<int64_t> layout;
    std::vector<BranchPageSnapshot> pages;
    ConversationCheckpoint running;
    std::vector<float> logits;
    std::vector<VisionPlacement> input_images;
};
BranchSnapshotCounters BranchSession::snapshot_counters() {
    std::lock_guard<std::mutex> lock(snapshot_mutex);
    return snapshot_work;
}
struct BranchSession::Binding {
    QsaState* state = nullptr;
    size_t part = 0;
    void* original = nullptr;
    uint64_t bytes = 0, per_cell = 0;
    BranchPages memory;
};

BranchSession::BranchSession(SessionState& session, const ModelGeometry& geometry, std::string contract)
    : session_(session), geometry_(geometry), contract_(std::move(contract)) {}
BranchSession::~BranchSession() {
    if (bindings_.empty()) return;
    if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
    for (const auto& b : bindings_) set_pointer(*b->state, b->part, b->original);
}
bool BranchSession::bind(std::string& error) {
    if (state_ != State::unbound || contract_.empty()) return fail(error, "invalid bind state or empty contract");
    const auto& g = geometry_;
    ConversationStateSizes sizes;
    if (!conversation_session_sizes(g, session_, sizes, error) || session_.max_cells <= 0 ||
        !session_.qsa_states || session_.qsa_alloc <= 0 || !sync(error)) return false;
    layout_ = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads,
        g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim, g.n_head, g.n_head_kv, g.head_dim,
        g.idx_q_heads, g.idx_key_dim, g.hc, g.hc_lr, g.n_expert, g.n_ff,
        session_.max_cells, session_.layer_lo, session_.layer_hi, session_.qsa_ord0, session_.qsa_alloc};
    std::vector<std::unique_ptr<Binding>> staged;
    try {
        for (int64_t i = 0; i < session_.qsa_alloc; ++i) {
            auto& q = session_.qsa_states[session_.qsa_ord0 + i];
            if (q.kv_mode != 0 || q.kv_elastic >= 0 || q.n_pages != q.n_slots)
                return fail(error, "hot pages require fully resident, identity-mapped QSA pools");
            std::vector<int32_t> table((size_t)q.n_pages);
            if (cudaMemcpy(table.data(), q.page_table, table.size()*4, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail(error, "cannot validate attention page table");
            for (size_t p = 0; p < table.size(); ++p)
                if (table[p] != (int32_t)p) return fail(error, "attention page table is not the identity");
            std::array<uint64_t, 5> parts{};
            if (!conversation_kv_part_sizes(q, g, q.max_cells, true, parts, error)) return false;
            parts[4] = (uint64_t)q.idx_pooled_rows * g.idx_key_dim * sizeof(float);
            layout_.insert(layout_.end(), {q.kv_int8, q.kv_q4, q.kv_hybrid, q.kv_rot,
                q.max_cells, q.n_pages, q.n_slots, q.idx_pooled_rows});
            const uint64_t cells = (uint64_t)q.n_pages * strata::kernels::qsa_real_shapes().page_size;
            for (size_t p = 0; p < parts.size(); ++p) {
                if (!parts[p]) continue;
                auto b = std::make_unique<Binding>();
                b->state = &q; b->part = p; b->original = pool_pointer(q, p); b->bytes = parts[p];
                b->per_cell = p == 4 ? (uint64_t)g.idx_key_dim * sizeof(float) : parts[p] / cells;
                if (!b->original || !b->memory.reserve(parts[p]))
                    return fail(error, "cannot allocate private session mapping");
                // Bind is required on an initialized empty session before any
                // graph capture or consumed state. Reserve stable virtual
                // capacity without copying unused max-capacity KV pages.
                staged.push_back(std::move(b));
            }
        }
    } catch (const std::exception& e) { return fail(error, e.what()); }
    uint64_t pages=0;
    const auto granularity=vmm_granularity();
    for (const auto& b:staged) {
        const auto n=b->bytes/granularity+(b->bytes%granularity!=0);
        if (n>std::numeric_limits<uint64_t>::max()-pages) return fail(error, "integrity capacity overflow");
        pages+=n;
    }
    if (!integrity_.reserve_pages(pages,granularity)) return fail(error, "cannot reserve all-pool integrity scratch");
    bindings_ = std::move(staged);
    for (const auto& b : bindings_) set_pointer(*b->state, b->part, b->memory.data());
    state_ = State::ready;
    return true;
}
bool BranchSession::begin(int64_t position, int64_t count, std::string& error) {
    if (state_ != State::ready || position != (int64_t)ids_.size() || count <= 0 ||
        position < 0 || position > session_.max_cells || count > session_.max_cells - position)
        return fail(error, "invalid consumed position, capacity, or transition");
    if (!sync(error)) { state_ = State::failed; return false; }
    const auto shape = strata::kernels::qsa_real_shapes();
    std::vector<BranchPages::Write> writes;
    try { writes.reserve(bindings_.size()); }
    catch (...) { state_ = State::failed; return fail(error, "cannot allocate write list"); }
    for (const auto& b : bindings_) {
        uint64_t first, end;
        if (b->part == 4) {
            first = position / shape.idx_block;
            end = (position + count) / shape.idx_block + 1; // includes the moving spare row
        } else {
            first = position / shape.page_size * shape.page_size;
            end = (position + count + shape.page_size - 1) / shape.page_size * shape.page_size;
        }
        writes.push_back({&b->memory,first*b->per_cell,(end-first)*b->per_cell});
    }
    if (!BranchPages::prepare_writes(writes,integrity_)) {
        state_ = State::failed;
        return fail(error, "cannot prepare private tail; restore a committed image before reuse");
    }
    pending_ = count;
    pending_input_ids_.clear(); pending_images_.clear();
    state_ = State::writing;
    return true;
}
bool BranchSession::install_positions(const VisionLayout& plan, std::string& error) {
    auto* table = const_cast<int32_t*>(strata::kernels::mrope_table());
    if (!table) return fail(error, "image positions require stable vision storage before graph capture");
    const size_t bytes = plan.positions.size() * sizeof(int32_t);
    if (!sync(error) || cudaMemcpy(table, plan.positions.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(error, "cannot install branch position table");
    counters_.position_uploaded_bytes += bytes;
    return true;
}
bool BranchSession::begin_inputs(const std::vector<int32_t>& tokens,
                                 const std::vector<VisionPlacement>& images,
                                 int64_t vocab, std::string& error) {
    if (state_ != State::ready || tokens.empty() || images.size() < input_images_.size() ||
        !std::equal(input_images_.begin(), input_images_.end(), images.begin()) ||
        (!logits_.empty() && logits_.size() != static_cast<uint64_t>(vocab)))
        return fail(error, "image append changes existing inputs or frontier vocabulary");
    for (size_t i = input_images_.size(); i < images.size(); ++i)
        if (images[i].start < static_cast<int64_t>(ids_.size()))
            return fail(error, "image append changes a consumed text input");
    try {
        auto next_ids = tokens;
        auto next_images = images;
        auto all = ids_;
        all.insert(all.end(), tokens.begin(), tokens.end());
        const auto plan = vision_layout(all, images, session_.max_cells, vocab, geometry_.n_embd, 1024);
        if (!strata::kernels::mrope_table()) return fail(error, "image append requires vision storage");
        if (!begin(static_cast<int64_t>(ids_.size()), static_cast<int64_t>(tokens.size()), error)) return false;
        if (!install_positions(plan, error)) { state_ = State::failed; return false; }
        pending_input_ids_.swap(next_ids); pending_images_.swap(next_images);
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::map_prefix(int64_t tokens, std::string& error) {
    if (tokens < 0 || tokens > session_.max_cells) return fail(error, "invalid prefix extent");
    for (const auto& binding : bindings_) {
        std::array<uint64_t, 5> sizes{};
        if (!conversation_kv_part_sizes(*binding->state, geometry_, tokens, true, sizes, error)) return false;
        if (sizes[binding->part] && !binding->memory.grow(sizes[binding->part]))
            return fail(error, "cannot map validated checkpoint prefix");
    }
    return true;
}
bool BranchSession::reset(std::string& error) {
    if (state_ == State::unbound) return fail(error, "cannot reset an unbound session");
    if (!sync(error)) { state_ = State::failed; return false; }
    state_ = State::failed;
    for (const auto& b : bindings_)
        if (!b->memory.clear()) return fail(error, "cannot clear private physical prefix");
    ConversationStateSizes sizes;
    if (!conversation_session_sizes(geometry_, session_, sizes, error)) return false;
    auto zero = [&](void* p, size_t bytes) {
        return !bytes || (p && cudaMemset(p, 0, bytes) == cudaSuccess);
    };
    if (!zero(session_.gdn_state, sizes.gdn) || !zero(session_.ple_hist, session_.ple_hist ? sizes.ple : 0))
        return fail(error, "cannot clear recurrent state");
    for (int64_t i = 0; i < session_.qsa_alloc; ++i) {
        const auto& q = session_.qsa_states[session_.qsa_ord0 + i];
        if (!zero(q.idx_tail, sizes.tail) || !zero(q.idx_dead, sizes.dead) || !zero(q.idx_block_pos, sizes.block_pos))
            return fail(error, "cannot clear indexer state");
    }
    if (!sync(error)) return false;
    session_.ple_prev[0] = session_.ple_prev[1] = session_.ple_token = -1;
    if (strata::kernels::mrope_table()) {
        try {
            const auto identity = vision_layout({}, {}, session_.max_cells,
                std::numeric_limits<int32_t>::max(), geometry_.n_embd, 1024);
            if (!install_positions(identity, error)) return false;
        } catch (const std::exception& e) { return fail(error, e.what()); }
    }
    ids_.clear(); logits_.clear(); input_images_.clear(); pending_images_.clear(); pending_input_ids_.clear();
    pending_ = 0; state_ = State::ready;
    return true;
}
bool BranchSession::finish(const std::vector<int32_t>& tokens, const std::vector<float>& logits, std::string& error) {
    if (state_ != State::writing || (int64_t)tokens.size() != pending_ || logits.empty() ||
        (!pending_input_ids_.empty() && pending_input_ids_ != tokens) ||
        std::any_of(tokens.begin(), tokens.end(), [](int32_t t) { return t < 0; }) ||
        std::any_of(logits.begin(), logits.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid consumed token or frontier result");
    if (!sync(error)) { state_ = State::failed; return false; }
    try {
        auto ids = ids_;
        ids.insert(ids.end(), tokens.begin(), tokens.end());
        auto next = logits;
        ids_.swap(ids); logits_.swap(next);
        if (!pending_input_ids_.empty()) input_images_.swap(pending_images_);
        pending_input_ids_.clear(); pending_images_.clear();
    } catch (const std::exception& e) { state_ = State::failed; return fail(error, e.what()); }
    state_ = State::ready;
    pending_ = 0;
    return true;
}
BranchSessionImage BranchSession::freeze(std::string& error) {
    if (state_ != State::ready || ids_.empty() || !sync(error)) {
        fail(error, "only a consumed, synchronized frontier can be captured");
        return {};
    }
    try {
        auto image = std::make_shared<BranchSessionSnapshot>();
        image->contract = contract_; image->layout = layout_;
        image->running.ids = ids_; image->logits = logits_;
        image->input_images = input_images_;
        if (!conversation_checkpoint_save(image->running, session_, geometry_, error)) return {};
        counters_.saved_running_bytes += running_bytes(image->running);
        for (const auto& b : bindings_) {
            auto pages = b->memory.freeze();
            if (!pages) { fail(error, "cannot protect captured pages"); return {}; }
            image->pages.push_back(std::move(pages));
        }
        uint64_t input_bytes = image->input_images.capacity() * sizeof(VisionPlacement);
        for (const auto& input : image->input_images) input_bytes += input.embedding_sha256.capacity() + 1;
        image->lifetime.publish(image->running.bytes() + image->logits.capacity() * sizeof(float) + input_bytes);
        ++counters_.captures;
        return image;
    } catch (const std::exception& e) { fail(error, e.what()); return {}; }
}
bool BranchSession::restore(const BranchSessionImage& image, std::string& error) {
    if (state_ == State::unbound || !image || image->contract != contract_ || image->layout != layout_ ||
        image->pages.size() != bindings_.size()) return fail(error, "checkpoint contract or layout differs");
    try {
        auto next_images = image->input_images;
        const bool has_table = strata::kernels::mrope_table() != nullptr;
        if (!next_images.empty() && !has_table) return fail(error, "image restore requires vision storage");
        VisionLayout positions;
        if (has_table)
            positions = vision_layout(image->running.ids, next_images, session_.max_cells,
                                      image->logits.size(), geometry_.n_embd, 1024);
        if (!conversation_checkpoint_validate(image->running, session_, geometry_, error) || !sync(error)) return false;
        state_ = State::failed; // Any partial restore is private and cannot be frozen.
        if (has_table && !install_positions(positions, error)) return false;
        for (size_t i = 0; i < bindings_.size(); ++i)
            if (!bindings_[i]->memory.restore(image->pages[i])) return fail(error, "cannot activate checkpoint pages");
        // The pooled spare row is already in the frozen page image. Restoring it a
        // second time would write a shared, read-only page for no semantic change.
        if (!conversation_checkpoint_restore_running(image->running, session_, geometry_, error)) return false;
        counters_.restored_running_bytes += running_bytes(image->running);
        ids_ = image->running.ids; logits_ = image->logits;
        input_images_.swap(next_images); pending_images_.clear(); pending_input_ids_.clear();
        pending_ = 0; state_ = State::ready; ++counters_.restores;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::write(const std::string& path, size_t& bytes, std::string& error) {
    bytes = 0;
    if (state_ != State::ready || ids_.empty() || !sync(error))
        return fail(error, "cold write requires a consumed, synchronized frontier");
    try {
        const auto images = conversation_images(input_images_);
        const std::vector<ConversationCheckpoint> checkpoints;
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        const ConversationView view{ids_, images, checkpoints, false};
        if (!conversation_snapshot_sources(meta, sources, view, session_, geometry_, nullptr, error) ||
            !session_file_write(path, meta, sources, file_identity(contract_, layout_), bytes, error)) return false;
        counters_.written_file_bytes += bytes;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::host_capture_bytes(size_t& bytes, std::string& error) const {
    if (!input_images_.empty()) return fail(error, "host image checkpoint bindings are not implemented");
    if (state_ != State::ready || ids_.empty() || logits_.empty())
        return fail(error, "host capture requires a consumed frontier");
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    if (!conversation_snapshot_bytes({ids_, images, checkpoints, false}, session_, geometry_, nullptr, bytes, error))
        return false;
    if (logits_.size() > (SIZE_MAX - bytes) / sizeof(float)) return fail(error, "host frontier size overflow");
    bytes += logits_.size() * sizeof(float);
    return true;
}
bool BranchSession::capture_host(SavedConversation& image, std::string& error) {
    size_t bytes = 0;
    if (!host_capture_bytes(bytes, error)) return false;
    try {
        const std::vector<ConversationImageKey> images;
        const std::vector<ConversationCheckpoint> checkpoints;
        SavedConversation captured;
        if (!conversation_snapshot_save(captured, {ids_, images, checkpoints, false},
                                        session_, geometry_, nullptr, error)) return false;
        captured.next_logits = logits_;
        image = std::move(captured);
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::restore_host(const SavedConversation& image, const std::vector<int32_t>& expected_ids,
                                const std::vector<float>& frontier, std::string& error) {
    if (state_ == State::unbound || expected_ids.empty() || expected_ids.size() > (size_t)session_.max_cells ||
        image.live.ids != expected_ids || !image.live.imgs.empty() || !image.checkpoints.empty() || image.cvec ||
        frontier.empty() || std::any_of(frontier.begin(), frontier.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid host checkpoint history or frontier");
    if (!conversation_snapshot_validate(image, session_, geometry_, nullptr, error)) return false;
    try {
        auto next_ids = expected_ids;
        auto next_logits = frontier;
        if (!reset(error)) return false;
        state_ = State::failed;
        if (!map_prefix((int64_t)expected_ids.size(), error)) return false;
        if (conversation_snapshot_restore(image, session_, geometry_, nullptr, error) != ConversationRestore::restored ||
            !sync(error)) return false;
        ids_.swap(next_ids); logits_.swap(next_logits);
        pending_ = 0; state_ = State::ready;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::load(const std::string& path, const std::vector<int32_t>& expected_ids,
                         const std::vector<float>& frontier, std::string& error) {
    return load_inputs(path, expected_ids, {}, frontier, error);
}
bool BranchSession::load_inputs(const std::string& path, const std::vector<int32_t>& expected_ids,
                                const std::vector<VisionPlacement>& expected_images,
                                const std::vector<float>& frontier, std::string& error) {
    if (state_ == State::unbound || expected_ids.empty() || expected_ids.size() > (size_t)session_.max_cells ||
        std::any_of(expected_ids.begin(), expected_ids.end(), [](int32_t t) { return t < 0; }) ||
        frontier.empty() || std::any_of(frontier.begin(), frontier.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid cold history or frontier");
    try {
        auto next_images = expected_images;
        const bool has_table = strata::kernels::mrope_table() != nullptr;
        if (!next_images.empty() && !has_table) return fail(error, "cold image restore requires vision storage");
        VisionLayout positions;
        if (has_table)
            positions = vision_layout(expected_ids, next_images, session_.max_cells,
                                      frontier.size(), geometry_.n_embd, 1024);
        const auto expected_keys = conversation_images(next_images);
        SessionReadLimits limits;
        if (!conversation_session_read_limits(limits, session_, geometry_, nullptr, session_.max_cells, 0, error))
            return false;
        auto next_logits = frontier; // Allocate before any private state is reset.
        SavedConversation image;
        size_t bytes = 0;
        std::vector<ConversationKvTarget> targets;
        bool prepared = false;
        const auto before_copy = conversation_detail::completed_copies;
        const SessionRestorePrepare prepare = [&](const SavedConversation& parsed,
                                                   const std::vector<std::array<uint64_t, 5>>& part_sizes,
                                                   std::string& why) {
            // Everything that can reject the checkpoint is checked before reset performs any device writes.
            if (parsed.live.ids != expected_ids || parsed.live.imgs != expected_keys || !parsed.checkpoints.empty() || parsed.cvec) {
                why = "cold checkpoint history differs"; return false;
            }
            if (!conversation_snapshot_validate_meta(parsed, session_, geometry_, nullptr, why)) return false;
            for (size_t layer = 0; layer < parsed.kv.size(); ++layer) {
                if (layer >= (size_t)session_.qsa_alloc) { why = "cold checkpoint contains an unsupported layer"; return false; }
                const QsaState& q = session_.qsa_states[session_.qsa_ord0 + layer];
                std::array<uint64_t, 5> expected{};
                if (!conversation_kv_part_sizes(q, geometry_, (int64_t)expected_ids.size(), true, expected, why)) return false;
                if (part_sizes[layer] != expected) { why = "cold checkpoint K/V part sizes differ"; return false; }
            }
            if (!reset(why)) return false;
            state_ = State::failed; // any failure after private reset/apply is fail-stop
            prepared = true;
            if (has_table && !install_positions(positions, why)) return false;
            // Metadata and exact extents are checked before mapping destinations;
            // the streaming callback starts only after the full set is mapped.
            for (size_t layer = 0; layer < part_sizes.size(); ++layer) {
                if (layer >= (size_t)session_.qsa_alloc) { why = "cold checkpoint contains an unsupported layer"; return false; }
                for (const auto& binding : bindings_) {
                    if (binding->state != &session_.qsa_states[session_.qsa_ord0 + layer]) continue;
                    if (part_sizes[layer][binding->part] &&
                        !binding->memory.grow(part_sizes[layer][binding->part])) {
                        why = "cannot map validated cold prefix"; return false;
                    }
                }
            }
            if (!conversation_kv_targets(parsed, session_, geometry_, (const QsaState*)nullptr, targets, why)) return false;
            for (size_t layer = 0; layer < targets.size(); ++layer)
                if (targets[layer].sizes != part_sizes[layer]) { why = "cold checkpoint target sizes differ"; return false; }
            return true;
        };
        const SessionRestoreApply apply = [&](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
            if (layer >= targets.size()) { error = "branch session: K/V layer outside validated targets"; return false; }
            return targets[layer].apply(part, offset, data, n, error);
        };
        if (!session_file_restore_streamed(path, file_identity(contract_, layout_), limits, prepare, apply,
                                           image, bytes, error)) {
            counters_.cold_restore_copy_bytes += conversation_detail::completed_copies.bytes-before_copy.bytes;
            counters_.cold_restore_copy_calls += conversation_detail::completed_copies.calls-before_copy.calls;
            if (prepared) state_ = State::failed;
            return false;
        }
        counters_.cold_restore_copy_bytes += conversation_detail::completed_copies.bytes-before_copy.bytes;
        counters_.cold_restore_copy_calls += conversation_detail::completed_copies.calls-before_copy.calls;
        for (const auto& target : targets) if (!target.finish(error)) { state_ = State::failed; return false; }
        if (!conversation_checkpoint_restore_running(image.live, session_, geometry_, error) || !sync(error)) {
            state_ = State::failed; return false;
        }
        ids_.swap(image.live.ids); logits_.swap(next_logits);
        input_images_.swap(next_images); pending_images_.clear(); pending_input_ids_.clear();
        pending_ = 0; state_ = State::ready;
        counters_.loaded_file_bytes += bytes;
        return true;
    } catch (const std::exception& e) {
        if (state_ != State::ready) state_ = State::failed;
        return fail(error, e.what());
    }
}
std::vector<BranchSessionPool> BranchSession::pools() const {
    std::vector<BranchSessionPool> out;
    for (const auto& b : bindings_) out.push_back({b->memory.data(), b->bytes, b->per_cell, b->part == 4});
    return out;
}
std::vector<BranchPageInfo> BranchSession::ownership() const {
    std::vector<BranchPageInfo> out;
    for (size_t i = 0; i < bindings_.size(); ++i) {
        const auto& b = bindings_[i];
        for (auto page : b->memory.ownership()) {
            page.kind = pool_kinds[b->part];
            page.pool = (int64_t)i;
            out.push_back(page);
        }
    }
    return out;
}
bool BranchSession::audit(const BranchSessionImage& image, const std::string& path,
                          BranchSessionAudit& report, std::string& error) const {
    if (!image || image->contract != contract_ || image->layout != layout_ ||
        image->pages.size() != bindings_.size()) return fail(error, "invalid audit image");
    if (!sync(error)) return false;
    try {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) return fail(error, "cannot open audit output");
        BranchSessionAudit result;
        result.consumed = image->running.ids.size();
        const auto shape = strata::kernels::qsa_real_shapes();
        result.qsa_page_cells = shape.page_size; result.index_block_cells = shape.idx_block;
        auto append = [&](BranchAuditRegion region, const void* data, size_t bytes) {
            if (!bytes) return true;
            region.offset = result.bytes; region.bytes = bytes;
            output.write((const char*)data, (std::streamsize)bytes);
            if (!output) return false;
            result.bytes += bytes;
            result.regions.push_back(std::move(region));
            return true;
        };
        for (size_t i = 0; i < bindings_.size(); ++i) {
            const auto& binding = bindings_[i];
            const auto pages = BranchPages::ownership(image->pages[i]);
            uint64_t pool_offset = 0;
            const auto complete = binding->part == 4 ? result.consumed / shape.idx_block :
                result.consumed / shape.page_size * shape.page_size;
            const auto filled = complete * binding->per_cell;
            for (size_t p = 0; p < pages.size(); ++p) {
                std::vector<uint8_t> bytes;
                if (!BranchPages::read_page(image->pages[i], p, bytes)) return fail(error, "cannot read saved physical page");
                BranchAuditRegion region;
                region.kind = pool_kinds[binding->part]; region.pool = (int64_t)i;
                region.qsa_ordinal = binding->state - session_.qsa_states;
                region.pool_offset = pool_offset; region.bytes_per_cell = binding->per_cell;
                region.filled_prefix_bytes = pool_offset < filled ? std::min(pages[p].bytes, filled - pool_offset) : 0;
                region.page = pages[p];
                if (!append(region, bytes.data(), bytes.size())) return fail(error, "cannot write page audit bytes");
                pool_offset += pages[p].bytes;
            }
        }
        auto field = [&](const char* kind, const auto& values) {
            BranchAuditRegion region; region.kind = kind;
            return append(region, values.data(), values.size() * sizeof(values[0]));
        };
        const auto& running = image->running;
        if (!field("gdn", running.gdn) || !field("ple", running.ple) || !field("tails", running.tails) ||
            !field("dead", running.dead) || !field("block_pos", running.block_pos) ||
            !field("ids", running.ids) || !field("logits", image->logits)) return fail(error, "cannot write running audit bytes");
        for (const auto& input : image->input_images) {
            const std::vector<int64_t> dimensions{input.start, input.rows, input.grid_width,
                input.grid_height, input.width, input.pad_token};
            if (!field("image_placement", dimensions) || !field("image_sha256", input.embedding_sha256))
                return fail(error, "cannot write image metadata audit bytes");
        }
        output.close();
        if (!output) return fail(error, "incomplete audit output");
        report = std::move(result);
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
} // namespace strata::core
