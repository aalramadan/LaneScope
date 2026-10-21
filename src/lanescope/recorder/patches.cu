#include "record.h"

using namespace lanescope;

namespace {

// Return values and access flags of the Sanitizer patching API (sanitizer_patching.h).
enum PatchResult : int { kPatchSuccess = 0 };
constexpr u32 kFlagRead = 0x1;
constexpr u32 kFlagWrite = 0x2;
constexpr u32 kFlagPrefetch = 0x8;

__device__ u32 linear_cta() {
    return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__device__ u32 linear_thread() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

// The header marks an atomic by both flags, or by a write that comes without its value.
__device__ u8 kind_of(u32 flags, bool value_known) {
    const bool reads = flags & kFlagRead;
    const bool writes = flags & kFlagWrite;
    if (writes && (reads || !value_known)) {
        return kAtomic;
    }
    return writes ? kStore : kLoad;
}

__device__ PatchResult record(void* userdata, u64 pc, u64 address, u32 width, u8 space, u8 kind) {
    Buffer* buffer = static_cast<Buffer*>(userdata);
    if (buffer == nullptr) {
        return kPatchSuccess;
    }
    // Every thread of a warp sees the same CTA, so the warp leaves together.
    const u32 cta = linear_cta();
    if (cta - buffer->cta_first >= buffer->cta_count) {
        return kPatchSuccess;
    }
    const u32 mask = __activemask();
    const u32 thread = linear_thread();
    const u32 lane = thread % kWarpSize;
    const u32 leader = __ffs(mask) - 1;
    u32 slot = 0;
    if (lane == leader) {
        slot = atomicAdd(&buffer->count, 1u);
    }
    slot = __shfl_sync(mask, slot, leader);
    if (slot >= buffer->capacity) {
        if (lane == leader) {
            atomicAdd(&buffer->dropped, 1u);
        }
        return kPatchSuccess;
    }
    Record& out = buffer->records[slot];
    out.addr[lane] = address;
    if (lane == leader) {
        out.pc = pc;
        out.cta = cta;
        out.warp = thread / kWarpSize;
        out.mask = mask;
        out.width = static_cast<u16>(width);
        out.space = space;
        out.kind = kind;
    }
    return kPatchSuccess;
}

}  // namespace

extern "C" __device__ __noinline__ PatchResult LaneScopeGlobalAccess(
    void* userdata, u64 pc, void* ptr, u32 accessSize, u32 flags, const void* data) {
    if (flags & kFlagPrefetch) {
        return kPatchSuccess;
    }
    const u8 kind = kind_of(flags, data != nullptr);
    return record(userdata, pc, reinterpret_cast<u64>(ptr), accessSize, kGlobal, kind);
}

extern "C" __device__ __noinline__ PatchResult LaneScopeSharedAccess(
    void* userdata, u64 pc, void* ptr, u32 accessSize, u32 flags, const void* data) {
    // For shared memory, ptr holds the offset within the shared window.
    const u8 kind = kind_of(flags, data != nullptr);
    return record(userdata, pc, reinterpret_cast<u64>(ptr), accessSize, kShared, kind);
}

// A cp.async copy reads global memory and writes shared memory at one PC, so it gives
// one record per side. A null source marks a copy that only fills zeros.
extern "C" __device__ __noinline__ PatchResult LaneScopeMemcpyAsync(
    void* userdata, u64 pc, void* src, u32 dst, u32 accessSize) {
    if (src != nullptr) {
        record(userdata, pc, reinterpret_cast<u64>(src), accessSize, kGlobal, kLoad);
    }
    return record(userdata, pc, dst, accessSize, kShared, kStore);
}
