// Every thread adds its values to bins of a shared histogram. The bins follow the
// input values, therefore the conflicts change from warp to warp.
extern "C" __global__ void histogram(const unsigned* values, unsigned* out, int n) {
    __shared__ unsigned bins[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) bins[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        atomicAdd(&bins[values[i] & 1023u], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) atomicAdd(&out[i], bins[i]);
}

// A tile read down its columns, the 32-way conflict that a padding or a swizzle removes.
// With TILE_PAD=1, each row holds one float more, and the threads of a warp read 32 banks.
#ifndef TILE_PAD
#define TILE_PAD 0
#endif
extern "C" __global__ void transpose(const float* in, float* out, int width) {
    __shared__ float tile[32][32 + TILE_PAD];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}

// A tree sum with interleaved addressing. Its stride doubles each step, which sets
// the conflicts of each step whatever the values.
extern "C" __global__ void tree_sum(const float* in, float* out) {
    __shared__ float data[256];
    unsigned t = threadIdx.x;
    data[t] = in[blockIdx.x * 256 + t];
    __syncthreads();
    for (unsigned s = 1; s < 256; s *= 2) {
        unsigned index = 2 * s * t;
        if (index < 256) data[index] += data[index + s];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = data[0];
}

// Neighboring threads read neighboring words, which takes no extra pass.
extern "C" __global__ void scale(const float* in, float* out, float factor) {
    __shared__ float data[256];
    unsigned t = threadIdx.x;
    data[t] = in[blockIdx.x * 256 + t];
    __syncthreads();
    out[blockIdx.x * 256 + t] = data[255 - t] * factor;
}
