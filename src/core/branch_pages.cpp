#include "strata/core/branch_pages.hpp"
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
#include "strata/core/branch_gpu_compat.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <new>

namespace strata::core {
namespace {
struct Counters {
    std::atomic<uint64_t> next_id{1},allocations{0},releases{0},live_bytes{0},cow_pages{0},copied_bytes{0};
    std::atomic<uint64_t> audit_read_bytes{0};
    std::atomic<uint64_t> integrity_checked_pages{0}, integrity_mismatches{0};
    std::atomic<uint64_t> last_integrity_mismatch_tile{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> seal_allocations{0}, seal_releases{0}, seal_live_bytes{0}, seal_upload_bytes{0};
};
Counters& counts() { static auto* c=new Counters; return *c; }
// Bound integrity scratch independently of context length. Consecutive sealed
// pages share one launch; the same ordered SHA-256 tiles still cover every byte.
constexpr uint64_t verification_batch_pages = 16;
struct PhysicalPage {
    VmmChunk handle=0;
    uint64_t id=0,bytes=0;
    int device=-1;
    // Ordered SHA-256 digests of fixed 4096-byte tiles. Empty means private,
    // writable state that has not yet been published as a numerical image.
    std::vector<BranchDigest> seal;
    // Device-side expected digests are tied to this physical allocation, never
    // to a branch/table address. They are only used for fresh-hash comparison.
    BranchDigest* device_seal=nullptr;
    uint64_t device_seal_bytes=0;
    bool device_seal_valid=false;
    PhysicalPage() {
        handle=vmm_chunk_new();
        if (!handle) throw std::bad_alloc();
        bytes=vmm_granularity();
        device=vmm_chunk_device(handle);
        auto& c=counts();
        id=c.next_id.fetch_add(1);
        ++c.allocations;
        c.live_bytes+=bytes;
    }
    ~PhysicalPage() {
        int previous=-1;
        if (cudaGetDevice(&previous)!=cudaSuccess ||
            (previous!=device && cudaSetDevice(device)!=cudaSuccess)) std::terminate();
        if (!vmm_chunk_release(handle)) std::terminate();
        if (device_seal) {
            if (cudaFree(device_seal)!=cudaSuccess) std::terminate();
            ++counts().seal_releases;
            counts().seal_live_bytes-=device_seal_bytes;
        }
        if (previous!=device && cudaSetDevice(previous)!=cudaSuccess) std::terminate();
        auto& c=counts();
        ++c.releases;
        c.live_bytes-=bytes;
    }
};
bool upload_device_seal(PhysicalPage& page) {
    if (page.seal.empty() || page.device!=vmm_current_device() ||
        page.seal.size()>std::numeric_limits<uint64_t>::max()/sizeof(BranchDigest)) return false;
    const uint64_t bytes=page.seal.size()*sizeof(BranchDigest);
    if (!page.device_seal) {
        if (cudaMalloc(&page.device_seal,bytes)!=cudaSuccess) return false;
        page.device_seal_bytes=bytes;
        ++counts().seal_allocations;
        counts().seal_live_bytes+=bytes;
    } else if (page.device_seal_bytes!=bytes) {
        return false;
    }
    if (cudaMemcpy(page.device_seal,page.seal.data(),bytes,cudaMemcpyHostToDevice)!=cudaSuccess) {
        page.device_seal_valid=false;
        return false;
    }
    page.device_seal_valid=true;
    counts().seal_upload_bytes+=bytes;
    return true;
}
}

struct BranchPageTable {
    uint64_t capacity=0;
    std::vector<std::shared_ptr<PhysicalPage>> pages;
};

BranchPages::BranchPages() = default;
BranchPages::~BranchPages() {
    if (mapping_.device()<0) { if (!release()) std::terminate(); return; }
    int previous=-1;
    if (cudaGetDevice(&previous)!=cudaSuccess) std::terminate();
    const int device=mapping_.device();
    if (device>=0 && previous!=device && cudaSetDevice(device)!=cudaSuccess) std::terminate();
    if (!release()) std::terminate();
    if (device>=0 && previous!=device && cudaSetDevice(previous)!=cudaSuccess) std::terminate();
}
bool BranchPages::release() {
    if (!current_owner()) return false;
    if (!verification_.release() || !mapping_.release()) return false;
    table_.reset();
    return true;
}
bool BranchPages::clear() {
    if (!current_owner() || !table_) return false;
    try {
        auto empty = std::make_shared<BranchPageTable>();
        empty->capacity = table_->capacity;
        if (!mapping_.clear()) return false;
        table_ = std::move(empty);
        return true;
    } catch (...) { return false; }
}
bool BranchPages::reserve(uint64_t bytes) {
    if (!current_owner()) return false;
    const auto g=vmm_granularity();
    if (!g || !bytes || bytes>std::numeric_limits<uint64_t>::max()-g+1 || !release() || !reserve_verification()) return false;
    try {
        auto table=std::make_shared<BranchPageTable>();
        table->capacity=((bytes+g-1)/g)*g;
        if (!mapping_.reserve(table->capacity)) return false;
        table_=std::move(table);
        return true;
    } catch (...) { return false; }
}
bool BranchPages::reserve_verification() {
    const auto g=vmm_granularity();
    return g && g%branch_digest_tile_bytes==0 && g<=std::numeric_limits<uint64_t>::max()/verification_batch_pages &&
        digest_.reserve(g*verification_batch_pages) && (verification_.base() || verification_.reserve(g));
}
bool BranchPages::private_table() {
    if (!table_) return false;
    if (!table_.unique()) {
        try { table_=std::make_shared<BranchPageTable>(*table_); }
        catch (...) { return false; }
    }
    return true;
}
bool BranchPages::grow(uint64_t bytes) {
    if (!current_owner() || !table_ || bytes>table_->capacity || !private_table()) return false;
    const auto g=vmm_granularity();
    const auto n=(bytes+g-1)/g;
    try {
        table_->pages.reserve(n);
        while (table_->pages.size()<n) {
            auto page=std::make_shared<PhysicalPage>();
            const auto i=table_->pages.size();
            if (!mapping_.map(i,page->handle,true)) return false;
            // Retain ownership even when zeroing fails; discard this private lease.
            table_->pages.push_back(std::move(page));
            if (cudaMemset(data()+i*g,0,g)!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) return false;
        }
        return true;
    } catch (...) { return false; }
}
BranchPageSnapshot BranchPages::freeze() {
    if (!current_owner() || !table_ || !verify_current()) return {};
    for (size_t i=0;i<table_->pages.size();++i) {
        auto& page=table_->pages[i];
        if (page->seal.empty()) {
            if (!digest_.digest(data()+i*page->bytes,page->bytes,page->seal)) return {};
            if (!upload_device_seal(*page)) { page->seal.clear(); return {}; }
        } else if (!page->device_seal_valid && !upload_device_seal(*page)) return {};
    }
    for (size_t i=0;i<table_->pages.size();++i) if (!mapping_.protect(i,false)) return {};
    return table_;
}
bool BranchPages::verify_current() {
    if (!table_) return false;
    for (size_t i=0;i<table_->pages.size();) {
        const auto& first=table_->pages[i];
        if (first->seal.empty()) { ++i; continue; }
        size_t end=i+1;
        while (end<table_->pages.size() && end-i<verification_batch_pages &&
               !table_->pages[end]->seal.empty()) ++end;
        std::vector<BranchDigest> actual;
        const auto per_page=first->bytes/branch_digest_tile_bytes;
        if (!digest_.digest(data()+i*first->bytes,(end-i)*first->bytes,actual) ||
            actual.size()!=(end-i)*per_page) return false;
        for (size_t page=i;page<end;++page) {
            const auto& seal=table_->pages[page]->seal;
            ++counts().integrity_checked_pages;
            if (seal.size()!=per_page || !std::equal(seal.begin(),seal.end(),actual.begin()+(page-i)*per_page)) {
                ++counts().integrity_mismatches; return false;
            }
        }
        i=end;
    }
    return true;
}
bool BranchPages::verify_snapshot(const BranchPageSnapshot& snapshot) {
    if (!snapshot || !reserve_verification()) return false;
    for (const auto& page:snapshot->pages) {
        if (page->seal.empty()) return false;
        std::vector<BranchDigest> actual;
        if (!verification_.map(0,page->handle,false)) return false;
        const bool complete=digest_.digest(verification_.base(),page->bytes,actual);
        if (!verification_.unmap(0) || !complete) return false;
        ++counts().integrity_checked_pages;
        if (actual!=page->seal) { ++counts().integrity_mismatches; return false; }
    }
    return true;
}
bool BranchPages::restore(const BranchPageSnapshot& snapshot) {
    if (!current_owner() || !snapshot ||
        (!snapshot->pages.empty() && snapshot->pages.front()->device!=vmm_current_device())) return false;
    // Validate the saved allocation before changing any active mappings.
    if (!verify_snapshot(snapshot)) return false;
    // Keep it alive even when restoring a snapshot of this range itself.
    std::shared_ptr<BranchPageTable> table;
    try { table=std::make_shared<BranchPageTable>(*snapshot); }
    catch (...) { return false; }
    if (mapping_.capacity()==table->capacity && mapping_.base()) {
        if (!mapping_.clear()) return false;
        table_.reset();
    } else if (!mapping_.release() || !mapping_.reserve(table->capacity)) return false;
    table_=std::move(table);
    for (size_t i=0;i<table_->pages.size();++i)
        if (!mapping_.map(i,table_->pages[i]->handle,false)) { release(); return false; }
    return true;
}
bool BranchPages::prepare_write(uint64_t offset,uint64_t bytes) {
    return valid_write(offset,bytes) && verify_current() && prepare_verified_write(offset,bytes);
}
bool BranchPages::valid_write(uint64_t offset,uint64_t bytes) const {
    if (!current_owner() || !table_) return false;
    const auto extent=table_->pages.size()*vmm_granularity();
    return offset<=extent && bytes<=extent-offset;
}
bool BranchPages::prepare_writes(const std::vector<Write>& writes, BranchDigestWorkspace& scratch) {
    // A small ordered list derived from the current page tables. Never cache a
    // successful hash: writable aliases can invalidate the previous result.
    // Check every pool before detaching ANY tail, including a corrupt last pool.
    const auto g=vmm_granularity();
    if (!g) return false;
    try {
        std::vector<const uint8_t*> inputs;
        std::vector<const BranchDigest*> seals;
        for (const auto& write:writes) {
            if (!write.pages || !write.pages->current_owner() || !write.pages->table_ ||
                write.offset>write.pages->table_->capacity ||
                write.bytes>write.pages->table_->capacity-write.offset) return false;
            const auto& pages=write.pages->table_->pages;
            for (size_t i=0;i<pages.size();++i) {
                if (pages[i]->seal.empty()) continue;
                if (pages[i]->device!=vmm_current_device() || !pages[i]->device_seal ||
                    !pages[i]->device_seal_valid || pages[i]->seal.size()!=g/branch_digest_tile_bytes) return false;
                inputs.push_back(write.pages->data()+i*g);
                seals.push_back(pages[i]->device_seal);
            }
        }
        if (!inputs.empty()) {
            uint64_t mismatch_tile=std::numeric_limits<uint64_t>::max();
            if (!scratch.compare_pages_view(inputs,seals,g,mismatch_tile)) return false;
            for (size_t i=0;i<seals.size();++i) ++counts().integrity_checked_pages;
            if (mismatch_tile!=std::numeric_limits<uint64_t>::max()) {
                ++counts().integrity_mismatches;
                counts().last_integrity_mismatch_tile=mismatch_tile;
                return false;
            }
        }
        // Verify every currently sealed page in every pool before mapping any
        // newly required physical extent. This keeps the all-pools-before-
        // mutation guarantee while allowing an empty reserved range to grow
        // only as its consumed prefix advances.
        for (const auto& write:writes)
            if (!write.pages->grow(write.offset+write.bytes)) return false;
        for (const auto& write:writes)
            if (!write.pages->prepare_verified_write(write.offset,write.bytes)) return false;
        return true;
    } catch (...) { return false; }
}
bool BranchPages::prepare_verified_write(uint64_t offset,uint64_t bytes) {
    if (!private_table()) return false;
    if (!bytes) return true;
    const auto g=vmm_granularity();
    const auto last=(offset+bytes-1)/g;
    try {
        for (auto i=offset/g;i<=last;++i) {
            if (table_->pages[i].use_count()==1) {
                if (!mapping_.protect(i,true)) return false;
                table_->pages[i]->seal.clear();
                table_->pages[i]->device_seal_valid=false;
                continue;
            }
            auto replacement=std::make_shared<PhysicalPage>();
            VmmMapping temporary;
            if (!temporary.reserve(g) || !temporary.map(0,replacement->handle,true)) return false;
            if (cudaMemcpy(temporary.base(),data()+i*g,g,cudaMemcpyDeviceToDevice)!=cudaSuccess ||
                cudaDeviceSynchronize()!=cudaSuccess) return false;
            ++counts().cow_pages;
            counts().copied_bytes+=g;
            if (!temporary.release()) std::terminate();
            if (!mapping_.unmap(i)) return false;
            if (!mapping_.map(i,replacement->handle,true)) { release(); return false; }
            table_->pages[i]=std::move(replacement);
        }
        return true;
    } catch (...) { return false; }
}
std::vector<BranchPageInfo> BranchPages::ownership() const {
    if (!current_owner()) return {};
    return ownership(table_);
}
std::vector<BranchPageInfo> BranchPages::ownership(const BranchPageSnapshot& snapshot) {
    std::vector<BranchPageInfo> out;
    if (snapshot) for (const auto& page:snapshot->pages)
        out.push_back({page->id,page->handle,(uint64_t)page.use_count(),page->bytes,nullptr,-1,page->device});
    return out;
}
bool BranchPages::read_page(const BranchPageSnapshot& snapshot, size_t index, std::vector<uint8_t>& bytes) {
    if (!snapshot || index >= snapshot->pages.size()) return false;
    const auto& page = snapshot->pages[index];
    if (page->device!=vmm_current_device()) return false;
    try {
        std::vector<uint8_t> result(page->bytes);
        VmmMapping borrow;
        if (!borrow.reserve(page->bytes) || !borrow.map(0, page->handle, false)) return false;
        if (cudaMemcpy(result.data(), borrow.base(), result.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        counts().audit_read_bytes += result.size();
        bytes.swap(result);
        return true;
    } catch (...) { return false; }
}
BranchPageCounters BranchPages::counters() {
    const auto& c=counts();
    const auto m=vmm_mapping_counters();
    return {c.allocations.load(),c.releases.load(),c.live_bytes.load(),c.cow_pages.load(),c.copied_bytes.load(),
        m.mappings,m.unmappings,m.reservations,m.address_releases,c.audit_read_bytes.load(),
        c.integrity_checked_pages.load(),c.integrity_mismatches.load(),c.last_integrity_mismatch_tile.load(),
        c.seal_allocations.load(),
        c.seal_releases.load(),c.seal_live_bytes.load(),c.seal_upload_bytes.load()};
}
} // namespace strata::core
