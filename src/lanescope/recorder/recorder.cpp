#include "record.h"

#include <dlfcn.h>
#include <sanitizer.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using lanescope::Buffer;
using lanescope::Record;
using lanescope::u32;
using lanescope::u64;

constexpr int kFormatVersion = 1;

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

// One recorded launch whose buffer waits for the next synchronization of its stream.
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
    void* device_memory = nullptr;
    // The asynchronous copy reads this header after the launch callback returns,
    // so the header lives on the heap until the flush.
    std::unique_ptr<Buffer> header;
};

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
        if (patched_contexts_.insert(data.context).second) {
            Check(sanitizerAddPatchesFromFile(settings_.patches.c_str(), data.context),
                  "loading the patches " + settings_.patches + " (build them for this GPU)");
        }
        const int index = next_module_++;
        modules_[data.module] = index;
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
        const size_t bytes = sizeof(Buffer) + static_cast<size_t>(settings_.capacity) * sizeof(Record);
        void* memory = nullptr;
        if (!Check(sanitizerAlloc(data.context, &memory, bytes), "allocating the record buffer")) {
            sanitizerSetCallbackData(data.function, nullptr);
            return;
        }
        Check(sanitizerMemset(memory, 0, bytes, data.hStream), "clearing the record buffer");
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
        launch.device_memory = memory;
        auto* records = reinterpret_cast<Record*>(static_cast<char*>(memory) + sizeof(Buffer));
        launch.header.reset(new Buffer{0, 0, settings_.capacity, settings_.cta_first,
                                       settings_.cta_count, 0, records});
        Check(sanitizerMemcpyHostToDeviceAsync(memory, launch.header.get(), sizeof(Buffer), data.hStream),
              "copying the buffer header");
        Check(sanitizerSetCallbackData(data.function, memory), "passing the buffer to the patches");
        pending_[{data.context, data.hStream}].push_back(std::move(launch));
    }

    void OnStreamSynchronized(CUcontext context, Sanitizer_StreamHandle stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        Flush({context, stream});
    }

    size_t PendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& entry : pending_) {
            count += entry.second.size();
        }
        return count;
    }

    // A later context may take the address of a destroyed one, and it needs the patches again.
    void OnContextDestroyed(CUcontext context) {
        OnContextSynchronized(context);
        std::lock_guard<std::mutex> lock(mutex_);
        patched_contexts_.erase(context);
    }

    void OnContextSynchronized(CUcontext context) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<CUcontext, Sanitizer_StreamHandle>> keys;
        for (const auto& entry : pending_) {
            if (entry.first.first == context) {
                keys.push_back(entry.first);
            }
        }
        for (const auto& key : keys) {
            Flush(key);
        }
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

    void Flush(const std::pair<CUcontext, Sanitizer_StreamHandle>& key) {
        auto found = pending_.find(key);
        if (found == pending_.end()) {
            return;
        }
        for (Launch& launch : found->second) {
            Save(launch, key.second);
        }
        pending_.erase(found);
    }

    void Save(Launch& launch, Sanitizer_StreamHandle stream) {
        Buffer header{};
        Check(sanitizerMemcpyDeviceToHost(&header, launch.device_memory, sizeof(header), stream),
              "copying the buffer header back");
        const u32 count = std::min(header.count, header.capacity);
        std::vector<Record> records(count);
        if (count > 0) {
            Check(sanitizerMemcpyDeviceToHost(records.data(), launch.header->records,
                                              count * sizeof(Record), stream),
                  "copying the records back");
        }
        const std::string stem = "launch-" + std::to_string(launch.index);
        WriteFile(Path(stem + ".bin"), records.data(), records.size() * sizeof(Record));
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
            ", \"dropped\": " + std::to_string(header.dropped) +
            ", \"capacity\": " + std::to_string(header.capacity) +
            ", \"cta_first\": " + std::to_string(header.cta_first) +
            ", \"cta_count\": " + std::to_string(header.cta_count) +
            ", \"record_bytes\": " + std::to_string(sizeof(Record)) + "}\n";
        WriteFile(Path(stem + ".json"), json.data(), json.size());
        Check(sanitizerFree(launch.context, launch.device_memory), "freeing the record buffer");
    }

    Settings settings_;
    std::mutex mutex_;
    std::set<CUcontext> patched_contexts_;
    std::map<CUmodule, int> modules_;
    std::map<std::pair<CUcontext, Sanitizer_StreamHandle>, std::vector<Launch>> pending_;
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
        // A program may destroy a stream without a synchronize, so its launches flush here.
        const auto& data = *static_cast<const Sanitizer_ResourceStreamData*>(cbdata);
        recorder->OnStreamSynchronized(data.context, data.hStream);
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
