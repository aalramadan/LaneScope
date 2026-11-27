#include <cupti.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace {

using u64 = unsigned long long;

constexpr int kFormatVersion = 1;
constexpr size_t kBufferBytes = size_t{8} << 20;

// Every kernel record since CUpti_ActivityKernel4 starts with the same fields up to the
// name, so this view reads the records of older and newer CUPTI versions alike.
using KernelRecord = CUpti_ActivityKernel4;

struct KernelTimes {
    u64 launches = 0;
    u64 nanoseconds = 0;
    u64 shortest = 0;
    u64 longest = 0;
};

std::mutex times_lock;
std::map<std::string, KernelTimes> kernels;
u64 skipped = 0;
u64 dropped = 0;
pid_t owner_pid = 0;

void Warn(const std::string& message) {
    std::fprintf(stderr, "lanescope kernel timer: %s\n", message.c_str());
}

std::string JsonText(const std::string& text) {
    std::string out = "\"";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = kBufferBytes;
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, kBufferBytes));
    *max_records = 0;
}

void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream, uint8_t* buffer, size_t,
                              size_t valid) {
    std::lock_guard<std::mutex> guard(times_lock);
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
        const auto* kernel = reinterpret_cast<const KernelRecord*>(record);
        if (kernel->end == 0 || kernel->end < kernel->start || kernel->name == nullptr) {
            ++skipped;
            continue;
        }
        const u64 duration = kernel->end - kernel->start;
        KernelTimes& times = kernels[kernel->name];
        times.shortest = times.launches ? std::min(times.shortest, duration) : duration;
        times.longest = std::max(times.longest, duration);
        times.nanoseconds += duration;
        ++times.launches;
    }
    size_t lost = 0;
    if (cuptiActivityGetNumDroppedRecords(context, stream, &lost) == CUPTI_SUCCESS) dropped += lost;
    std::free(buffer);
}

void WriteTimes() {
    // A forked child holds a copy of the times of its parent, which the parent writes.
    if (getpid() != owner_pid) return;
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    const char* base = std::getenv("LANESCOPE_KERNEL_TIMES");
    if (base == nullptr || *base == '\0') return;
    std::lock_guard<std::mutex> guard(times_lock);
    std::string text = "{\"format\": " + std::to_string(kFormatVersion) +
                       ", \"pid\": " + std::to_string(owner_pid) +
                       ", \"skipped\": " + std::to_string(skipped) +
                       ", \"dropped\": " + std::to_string(dropped) + ", \"kernels\": {";
    bool first = true;
    for (const auto& [name, times] : kernels) {
        text += (first ? "" : ", ") + JsonText(name) +
                ": {\"launches\": " + std::to_string(times.launches) +
                ", \"nanoseconds\": " + std::to_string(times.nanoseconds) +
                ", \"shortest\": " + std::to_string(times.shortest) +
                ", \"longest\": " + std::to_string(times.longest) + "}";
        first = false;
    }
    text += "}}\n";
    const std::string path = std::string(base) + "." + std::to_string(owner_pid);
    const std::string staged = path + ".part";
    FILE* file = std::fopen(staged.c_str(), "w");
    if (file == nullptr) {
        Warn("cannot write " + staged);
        return;
    }
    const bool written = std::fwrite(text.data(), 1, text.size(), file) == text.size();
    if (std::fclose(file) != 0 || !written || std::rename(staged.c_str(), path.c_str()) != 0) {
        Warn("cannot write " + path);
    }
}

}  // namespace

extern "C" int InitializeInjection(void) {
    owner_pid = getpid();
    const char* text = nullptr;
    CUptiResult result = cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    if (result == CUPTI_SUCCESS) result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
        cuptiGetResultString(result, &text);
        Warn(std::string("CUPTI cannot time kernels: ") + (text ? text : "unknown error"));
        return 0;
    }
    std::atexit(WriteTimes);
    return 1;
}
