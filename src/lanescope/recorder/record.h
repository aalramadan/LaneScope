#pragma once

namespace lanescope {

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

constexpr u32 kWarpSize = 32;

enum : u8 { kShared = 0, kGlobal = 1 };
enum : u8 { kLoad = 0, kStore = 1, kAtomic = 2 };

// One warp-level execution of a memory instruction. Inactive lanes keep address 0.
struct Record {
    u64 pc;
    u64 addr[kWarpSize];
    u32 cta;
    u32 warp;
    u32 mask;
    u16 width;
    u8 space;
    u8 kind;
};
static_assert(sizeof(Record) == 280, "the importer expects 280-byte records");

// Counts of one launch, in pinned host memory just before its records. The threads that
// have exited tell the host when the records are complete.
struct Counters {
    u32 count;
    u32 dropped;
    u64 exited;
};

// Header of one launch in device memory, which every patched access reads. The counters
// and the records live in pinned host memory, which the patches write through the same
// addresses, so the host reads a finished launch without a copy.
struct Buffer {
    u32 capacity;
    u32 cta_first;
    u32 cta_count;
    u32 reserved;
    Counters* counters;
    Record* records;
};
static_assert(sizeof(Buffer) == 32, "the host copies a 32-byte header");

}  // namespace lanescope
