#include "cubin.h"
#include "record.h"

#include <dlfcn.h>
#include <sanitizer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using lanescope::Buffer;
using lanescope::Counters;
using lanescope::Record;
using lanescope::u32;
using lanescope::u64;

constexpr int kFormatVersion = 1;
// Pinned host memory that saved buffers may keep for reuse.
constexpr size_t kPendingBytes = size_t{1} << 30;

void Warn(const std::string& message) {
    std::fprintf(stderr, "lanescope recorder: %s\n", message.c_str());
}

bool Check(SanitizerResult result, const std::string& action) {
    if (result == SANITIZER_SUCCESS) {
        return true;
    }
    const char* text = nullptr;
    sanitizerGetResultString(result, &text);
    Warn(action + " failed: " + (text ? text : "unknown error"));
    return false;
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

struct Settings {
    std::string out_dir;
    std::string patches;
    u32 capacity = 1u << 18;
    u32 cta_first = 0;
    u32 cta_count = 64;
    std::string kernel_filter;

    static Settings FromEnvironment() {
        Settings settings;
        if (const char* value = std::getenv("LANESCOPE_OUT")) {
            settings.out_dir = value;
        }
        if (const char* value = std::getenv("LANESCOPE_PATCHES")) {
            settings.patches = value;
        }
        if (const char* value = std::getenv("LANESCOPE_CAPACITY")) {
            char* end = nullptr;
            const unsigned long capacity = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0' && capacity >= 1 && capacity <= (1ul << 26)) {
                settings.capacity = static_cast<u32>(capacity);
            } else {
                Warn("LANESCOPE_CAPACITY needs a count from 1 to 2^26, keeping the default");
            }
        }
        if (const char* value = std::getenv("LANESCOPE_CTAS")) {
            unsigned first = 0;
            unsigned count = 0;
            if (std::sscanf(value, "%u:%u", &first, &count) == 2) {
                settings.cta_first = first;
                settings.cta_count = count;
            } else {
                Warn("LANESCOPE_CTAS needs first:count, keeping the default");
            }
        }
        if (const char* value = std::getenv("LANESCOPE_KERNEL")) {
            settings.kernel_filter = value;
        }
        return settings;
    }
};

// The pinned host memory of the counters and records of one launch, with the device header
// that points the patches at them. Saved launches give theirs back for reuse, since pinning
// memory takes far longer than a launch.
struct Slot {
    void* host = nullptr;
    void* device_header = nullptr;
    size_t bytes = 0;
};

// One recorded launch whose records wait until it has finished.
struct Launch {
    int index = 0;
    std::string kernel;
    int module_index = -1;
    uint64_t function_pc = 0;
    uint64_t function_size = 0;
    u32 grid[3] = {0, 0, 0};
    u32 block[3] = {0, 0, 0};
    int arch_major = 0;
    int arch_minor = 0;
    CUcontext context = nullptr;
    CUdevice device = 0;
    Slot slot;
    u64 threads = 0;
    // The asynchronous copy reads this header after the launch callback returns,
    // so the header lives on the heap until the flush.
    std::unique_ptr<Buffer> header;
};

size_t DeviceMemory(CUdevice device) {
    using TotalMemory = CUresult (*)(size_t*, CUdevice);
    static const auto get = [] {
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        void* symbol = driver ? dlsym(driver, "cuDeviceTotalMem_v2") : nullptr;
        return reinterpret_cast<TotalMemory>(symbol);
    }();
    size_t bytes = 0;
    if (get == nullptr || get(&bytes, device) != CUDA_SUCCESS) {
        return 0;
    }
    return bytes;
}

int DeviceAttribute(CUdevice_attribute attribute, CUdevice device) {
    // The application has loaded the CUDA driver already, so the tool needs no link to it.
    // A loader such as Python's ctypes keeps the driver out of the global scope, which
    // is why the lookup goes through a handle to the loaded library.
    using GetAttribute = CUresult (*)(int*, CUdevice_attribute, CUdevice);
    static const auto get = [] {
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        void* symbol = driver ? dlsym(driver, "cuDeviceGetAttribute") : nullptr;
        if (symbol == nullptr) {
            Warn("cannot find cuDeviceGetAttribute, so launch files name the architecture sm_00");
        }
        return reinterpret_cast<GetAttribute>(symbol);
    }();
    int value = 0;
    if (get == nullptr || get(&value, attribute, device) != CUDA_SUCCESS) {
        return 0;
    }
    return value;
}

class Recorder {
public:
    explicit Recorder(Settings settings) : settings_(std::move(settings)) {}

    void OnModuleLoaded(const Sanitizer_ResourceModuleData& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int index = next_module_++;
        modules_[data.module] = index;
        // Every patched instruction slows its kernel, even where nothing records, so a
        // module without a kernel to record keeps its own code.
        if (!settings_.kernel_filter.empty() &&
            !lanescope::NamesFunction(data.pCubin, data.cubinSize, settings_.kernel_filter)) {
            return;
        }
        if (patched_contexts_.insert(data.context).second) {
            Check(sanitizerAddPatchesFromFile(settings_.patches.c_str(), data.context),
                  "loading the patches " + settings_.patches + " (build them for this GPU)");
        }
        if (data.pCubin != nullptr && data.cubinSize > 0) {
            WriteFile(Path("module-" + std::to_string(index) + ".cubin"), data.pCubin, data.cubinSize);
        }
        Check(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, data.module,
                                         "LaneScopeGlobalAccess"),
              "patching global accesses");
        Check(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, data.module,
                                         "LaneScopeSharedAccess"),
              "patching shared accesses");
        Check(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMCPY_ASYNC, data.module,
                                         "LaneScopeMemcpyAsync"),
              "patching cp.async copies");
        Check(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MATRIX_MEMORY_ACCESS, data.module,
                                         "LaneScopeMatrixAccess"),
              "patching ldmatrix and stmatrix");
        Check(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, data.module, "LaneScopeBlockExit"),
              "patching block exits");
        Check(sanitizerPatchModule(data.module), "patching the module");
    }

    void OnLaunchBegin(const Sanitizer_LaunchData& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string kernel = data.functionName ? data.functionName : "";
        // A stale pointer from an earlier launch of the same kernel must never reach the patches.
        if (!settings_.kernel_filter.empty() && kernel.find(settings_.kernel_filter) == std::string::npos) {
            sanitizerSetCallbackData(data.function, nullptr);
            return;
        }
        SaveFinished(data.context);
        const size_t bytes = sizeof(Counters) + static_cast<size_t>(settings_.capacity) * sizeof(Record);
        // A launch that has not finished keeps its buffer. Launches past a quarter of the
        // memory of the device go unrecorded, which only a program whose launches run
        // behind its queue reaches.
        size_t& pending = pending_bytes_[data.device];
        if (pending > 0 && pending + bytes > DeviceMemory(data.device) / 4) {
            ++skipped_;
            sanitizerSetCallbackData(data.function, nullptr);
            return;
        }
        const Slot slot = TakeSlot(data.context, bytes);
        if (slot.host == nullptr) {
            sanitizerSetCallbackData(data.function, nullptr);
            return;
        }
        auto* counters = static_cast<Counters*>(slot.host);
        Launch launch;
        launch.index = next_launch_++;
        launch.kernel = kernel;
        launch.module_index = modules_.count(data.module) ? modules_[data.module] : -1;
        Check(sanitizerGetFunctionPcAndSize(data.module, data.functionName, &launch.function_pc,
                                            &launch.function_size),
              "finding the start PC of " + kernel);
        launch.grid[0] = data.gridDim_x;
        launch.grid[1] = data.gridDim_y;
        launch.grid[2] = data.gridDim_z;
        launch.block[0] = data.blockDim_x;
        launch.block[1] = data.blockDim_y;
        launch.block[2] = data.blockDim_z;
        launch.arch_major = DeviceAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, data.device);
        launch.arch_minor = DeviceAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, data.device);
        launch.context = data.context;
        launch.device = data.device;
        launch.slot = slot;
        pending += bytes;
        launch.header.reset(new Buffer{settings_.capacity, settings_.cta_first, settings_.cta_count, 0,
                                       counters, reinterpret_cast<Record*>(counters + 1)});
        Check(sanitizerMemcpyHostToDeviceAsync(slot.device_header, launch.header.get(), sizeof(Buffer),
                                               data.hStream),
              "copying the buffer header");
        Check(sanitizerSetCallbackData(data.function, slot.device_header),
              "passing the buffer to the patches");
        launch.threads = static_cast<u64>(data.gridDim_x) * data.gridDim_y * data.gridDim_z *
                         data.blockDim_x * data.blockDim_y * data.blockDim_z;
        pending_[{data.context, data.hStream}].push_back(std::move(launch));
    }

    void OnStreamSynchronized(CUcontext context, Sanitizer_StreamHandle stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = pending_.find({context, stream});
        if (found != pending_.end()) {
            SaveFinishedOf(found->second);
        }
    }

    int SkippedCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return skipped_;
    }

    size_t PendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& entry : pending_) {
            count += entry.second.size();
        }
        for (const auto& entry : orphans_) {
            count += entry.second.size();
        }
        return count;
    }

    // A later stream may take the handle of a destroyed one, so the launches of a destroyed
    // stream wait in a list of their own until all their threads have exited.
    void OnStreamDestroyed(CUcontext context, Sanitizer_StreamHandle stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = pending_.find({context, stream});
        if (found == pending_.end()) {
            return;
        }
        auto& orphans = orphans_[context];
        for (Launch& launch : found->second) {
            orphans.push_back(std::move(launch));
        }
        pending_.erase(found);
    }

    // A later context may take the address of a destroyed one, and it needs the patches again.
    void OnContextDestroyed(CUcontext context) {
        WaitForContext(context);
        OnContextSynchronized(context);
        std::lock_guard<std::mutex> lock(mutex_);
        patched_contexts_.erase(context);
        auto found = free_slots_.find(context);
        if (found != free_slots_.end()) {
            for (const Slot& slot : found->second) {
                free_bytes_ -= slot.bytes;
                FreeSlot(context, slot);
            }
            free_slots_.erase(found);
        }
    }

    void OnContextSynchronized(CUcontext context) {
        std::lock_guard<std::mutex> lock(mutex_);
        SaveFinished(context);
    }

    // A context that goes without a synchronization may still run some launches, so the
    // recorder waits a while for them to finish.
    void WaitForContext(CUcontext context) {
        for (int poll = 0; poll < 30000; ++poll) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                SaveFinished(context);
                if (!HasLaunches(context)) {
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Warn("some launches had not finished when their context went, so they have no launch files");
    }

private:
    std::string Path(const std::string& name) const { return settings_.out_dir + "/" + name; }

    static void WriteFile(const std::string& path, const void* data, size_t bytes) {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            Warn("cannot write " + path);
            return;
        }
        if (bytes > 0 && std::fwrite(data, 1, bytes, file) != bytes) {
            Warn("short write to " + path);
        }
        std::fclose(file);
    }

    // The threads of a launch count their exits after their records, so a launch whose every
    // thread has exited holds its complete records.
    static bool Finished(const Launch& launch) {
        const auto* counters = static_cast<const Counters*>(launch.slot.host);
        return __atomic_load_n(&counters->exited, __ATOMIC_ACQUIRE) >= launch.threads;
    }

    // Saves the finished launches of one stream in order. A launch that another thread queued
    // after the last synchronization may still run, and it waits for a later check.
    void SaveFinishedOf(std::deque<Launch>& launches) {
        while (!launches.empty() && Finished(launches.front())) {
            Save(launches.front());
            launches.pop_front();
        }
    }

    void SaveFinished(CUcontext context) {
        for (auto entry = pending_.begin(); entry != pending_.end();) {
            if (entry->first.first == context) {
                SaveFinishedOf(entry->second);
            }
            entry = entry->second.empty() ? pending_.erase(entry) : std::next(entry);
        }
        auto found = orphans_.find(context);
        if (found == orphans_.end()) {
            return;
        }
        auto& orphans = found->second;
        for (auto launch = orphans.begin(); launch != orphans.end();) {
            if (Finished(*launch)) {
                Save(*launch);
                launch = orphans.erase(launch);
            } else {
                ++launch;
            }
        }
        if (orphans.empty()) {
            orphans_.erase(found);
        }
    }

    bool HasLaunches(CUcontext context) const {
        for (const auto& entry : pending_) {
            if (entry.first.first == context && !entry.second.empty()) {
                return true;
            }
        }
        return orphans_.count(context) > 0;
    }

    Slot TakeSlot(CUcontext context, size_t bytes) {
        auto& kept = free_slots_[context];
        if (!kept.empty()) {
            const Slot slot = kept.back();
            kept.pop_back();
            free_bytes_ -= slot.bytes;
            return slot;
        }
        Slot slot;
        slot.bytes = bytes;
        if (!Check(sanitizerAllocHost(context, &slot.host, bytes), "allocating the record buffer")) {
            return Slot{};
        }
        if (!Check(sanitizerAlloc(context, &slot.device_header, sizeof(Buffer)),
                   "allocating the buffer header")) {
            sanitizerFreeHost(context, slot.host);
            return Slot{};
        }
        std::memset(slot.host, 0, bytes);
        return slot;
    }

    static void FreeSlot(CUcontext context, const Slot& slot) {
        Check(sanitizerFreeHost(context, slot.host), "freeing the record buffer");
        Check(sanitizerFree(context, slot.device_header), "freeing the buffer header");
    }

    // The launch has finished, so its counters and records in pinned host memory are
    // complete and need no copy.
    void Save(Launch& launch) {
        const Buffer& header = *launch.header;
        auto* counters = static_cast<Counters*>(launch.slot.host);
        const u32 count = std::min(counters->count, header.capacity);
        const auto* records = reinterpret_cast<const Record*>(counters + 1);
        const std::string stem = "launch-" + std::to_string(launch.index);
        WriteFile(Path(stem + ".bin"), records, count * sizeof(Record));
        const std::string json =
            "{\"format\": " + std::to_string(kFormatVersion) +
            ", \"kernel\": " + JsonText(launch.kernel) +
            ", \"launch\": " + std::to_string(launch.index) +
            ", \"module\": " + std::to_string(launch.module_index) +
            ", \"function_pc\": " + std::to_string(launch.function_pc) +
            ", \"function_size\": " + std::to_string(launch.function_size) +
            ", \"grid\": [" + std::to_string(launch.grid[0]) + ", " + std::to_string(launch.grid[1]) +
            ", " + std::to_string(launch.grid[2]) + "]" +
            ", \"block\": [" + std::to_string(launch.block[0]) + ", " + std::to_string(launch.block[1]) +
            ", " + std::to_string(launch.block[2]) + "]" +
            ", \"arch\": \"sm_" + std::to_string(launch.arch_major) + std::to_string(launch.arch_minor) + "\"" +
            ", \"records\": " + std::to_string(count) +
            ", \"dropped\": " + std::to_string(counters->dropped) +
            ", \"capacity\": " + std::to_string(header.capacity) +
            ", \"cta_first\": " + std::to_string(header.cta_first) +
            ", \"cta_count\": " + std::to_string(header.cta_count) +
            ", \"record_bytes\": " + std::to_string(sizeof(Record)) + "}\n";
        WriteFile(Path(stem + ".json"), json.data(), json.size());
        pending_bytes_[launch.device] -= launch.slot.bytes;
        // The next launch of the slot reads zeros wherever its warps write nothing.
        std::memset(counters, 0, sizeof(Counters) + count * sizeof(Record));
        // Every buffer of a recording has the same size, so a kept one serves any later launch.
        if (free_bytes_ + launch.slot.bytes <= kPendingBytes) {
            free_slots_[launch.context].push_back(launch.slot);
            free_bytes_ += launch.slot.bytes;
        } else {
            FreeSlot(launch.context, launch.slot);
        }
    }

    Settings settings_;
    std::mutex mutex_;
    std::set<CUcontext> patched_contexts_;
    std::map<CUmodule, int> modules_;
    std::map<std::pair<CUcontext, Sanitizer_StreamHandle>, std::deque<Launch>> pending_;
    std::map<CUcontext, std::deque<Launch>> orphans_;
    std::map<CUcontext, std::vector<Slot>> free_slots_;
    size_t free_bytes_ = 0;
    std::map<CUdevice, size_t> pending_bytes_;
    int skipped_ = 0;
    int next_module_ = 0;
    int next_launch_ = 0;
};

Recorder* active_recorder = nullptr;

void Callback(void* userdata, Sanitizer_CallbackDomain domain, Sanitizer_CallbackId cbid,
              const void* cbdata) {
    auto* recorder = static_cast<Recorder*>(userdata);
    if (domain == SANITIZER_CB_DOMAIN_RESOURCE && cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
        recorder->OnModuleLoaded(*static_cast<const Sanitizer_ResourceModuleData*>(cbdata));
    } else if (domain == SANITIZER_CB_DOMAIN_RESOURCE &&
               cbid == SANITIZER_CBID_RESOURCE_STREAM_DESTROY_STARTING) {
        const auto& data = *static_cast<const Sanitizer_ResourceStreamData*>(cbdata);
        recorder->OnStreamDestroyed(data.context, data.hStream);
    } else if (domain == SANITIZER_CB_DOMAIN_RESOURCE &&
               cbid == SANITIZER_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
        recorder->OnContextDestroyed(static_cast<const Sanitizer_ResourceContextData*>(cbdata)->context);
    } else if (domain == SANITIZER_CB_DOMAIN_LAUNCH && cbid == SANITIZER_CBID_LAUNCH_BEGIN) {
        recorder->OnLaunchBegin(*static_cast<const Sanitizer_LaunchData*>(cbdata));

    } else if (domain == SANITIZER_CB_DOMAIN_SYNCHRONIZE) {
        const auto& data = *static_cast<const Sanitizer_SynchronizeData*>(cbdata);
        if (cbid == SANITIZER_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED) {
            recorder->OnStreamSynchronized(data.context, data.hStream);
        } else if (cbid == SANITIZER_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED) {
            recorder->OnContextSynchronized(data.context);
        }
    }
}

void WarnAboutPendingLaunches() {
    const size_t pending = active_recorder ? active_recorder->PendingCount() : 0;
    if (pending > 0) {
        Warn(std::to_string(pending) +
             " launches were never synchronized before exit, so they have no launch files");
    }
    const int skipped = active_recorder ? active_recorder->SkippedCount() : 0;
    if (skipped > 0) {
        Warn(std::to_string(skipped) +
             " launches have no launch files, since the unfinished launches before them held a quarter of "
             "the GPU memory. A smaller --capacity records more of them");
    }
}

int Start() {
    Settings settings = Settings::FromEnvironment();
    if (settings.out_dir.empty() || settings.patches.empty()) {
        Warn("set LANESCOPE_OUT and LANESCOPE_PATCHES to record");
        return 1;
    }
    // The recorder lives as long as the process, so it is never freed.
    auto* recorder = new Recorder(std::move(settings));
    active_recorder = recorder;
    std::atexit(WarnAboutPendingLaunches);
    Sanitizer_SubscriberHandle handle;
    if (!Check(sanitizerSubscribe(&handle, Callback, recorder), "subscribing to callbacks")) {
        return 1;
    }
    Check(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE), "enabling resource callbacks");
    Check(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_LAUNCH), "enabling launch callbacks");
    Check(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE), "enabling synchronize callbacks");
    return 0;
}

// LD_PRELOAD runs this initializer before main, as in the MemoryTracker sample of NVIDIA.
[[maybe_unused]] const int started = Start();

}  // namespace
