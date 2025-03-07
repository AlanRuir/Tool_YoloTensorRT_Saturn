#include "kernel_function.cuh"

// 缩放图像核函数
__global__ void resizeKernel(int batch_size, unsigned char* src, int src_width, int src_height,
                             float* dst, int dst_width, int dst_height, float padding_value,
                             AffineMat dst2src)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dst_width * dst_height * 3;
    if (idx >= total)
        return;

    int channel = idx % 3;                              // 通道 (R/G/B)
    int dst_y   = (idx / 3) % dst_height;               // 目标高度
    int dst_x   = (idx / (3 * dst_height)) % dst_width; // 目标宽度
    int batch   = idx / (3 * dst_width * dst_height);   // 批次

    // 计算原始坐标
    float src_x = dst2src.v0_ * dst_x + dst2src.v1_ * dst_y + dst2src.v2_;
    float src_y = dst2src.v3_ * dst_x + dst2src.v4_ * dst_y + dst2src.v5_;

    // 检查边界
    if (src_x < 0 || src_x >= src_width - 1 || src_y < 0 || src_y >= src_height - 1)
    {
        dst[idx] = padding_value;
        return;
    }

    // 双线性插值
    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = src_x - x0;
    float wy = src_y - y0;

    int src_idx00 = (batch * src_height * src_width + y0 * src_width + x0) * 3 + channel;
    int src_idx01 = (batch * src_height * src_width + y0 * src_width + x1) * 3 + channel;
    int src_idx10 = (batch * src_height * src_width + y1 * src_width + x0) * 3 + channel;
    int src_idx11 = (batch * src_height * src_width + y1 * src_width + x1) * 3 + channel;

    float val = (1 - wx) * (1 - wy) * src[src_idx00] + wx * (1 - wy) * src[src_idx01] + (1 - wx) * wy * src[src_idx10] + wx * wy * src[src_idx11];
    dst[idx]  = val;
}

void resizeDevice(int batch_size, unsigned char* src, int src_width, int src_height,
                  float* dst, int dst_width, int dst_height, float padding_value,
                  AffineMat dst2src, cudaStream_t stream)
{
    int total             = batch_size * dst_width * dst_height * 3;
    int threads_per_block = 256;
    int blocks            = (total + threads_per_block - 1) / threads_per_block;
    resizeKernel<<<blocks, threads_per_block, 0, stream>>>(batch_size, src, src_width, src_height,
                                                           dst, dst_width, dst_height, padding_value, dst2src);
}

// BGR 转 RGB 核函数
__global__ void bgr2RgbKernel(int batch_size, float* src, int src_width, int src_height,
                              float* dst, int dst_width, int dst_height)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * src_width * src_height * 3;
    if (idx >= total)
        return;

    int channel = idx % 3;                              // 通道
    int src_y   = (idx / 3) % src_height;               // 高度
    int src_x   = (idx / (3 * src_height)) % src_width; // 宽度
    int batch   = idx / (3 * src_width * src_height);   // 批次

    int src_idx  = (batch * src_height * src_width + src_y * src_width + src_x) * 3;
    int dst_idx  = src_idx + (2 - channel); // BGR -> RGB: B(0)->R(2), G(1)->G(1), R(2)->B(0)
    dst[dst_idx] = src[src_idx + channel];
}

void bgr2RgbDevice(int batch_size, float* src, int src_width, int src_height,
                   float* dst, int dst_width, int dst_height, cudaStream_t stream)
{
    int total             = batch_size * src_width * src_height * 3;
    int threads_per_block = 256;
    int blocks            = (total + threads_per_block - 1) / threads_per_block;
    bgr2RgbKernel<<<blocks, threads_per_block, 0, stream>>>(batch_size, src, src_width, src_height,
                                                            dst, dst_width, dst_height);
}

// 归一化核函数
__global__ void normKernel(int batch_size, float* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height, float mean[3], float stds[3], float scale)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * src_width * src_height * 3;
    if (idx >= total)
        return;

    int channel = idx % 3;                              // 通道
    int src_y   = (idx / 3) % src_height;               // 高度
    int src_x   = (idx / (3 * src_height)) % src_width; // 宽度
    int batch   = idx / (3 * src_width * src_height);   // 批次

    int src_idx  = (batch * src_height * src_width + src_y * src_width + src_x) * 3 + channel;
    dst[src_idx] = (src[src_idx] / scale - mean[channel]) / stds[channel];
}

void normDevice(int batch_size, float* src, int src_width, int src_height,
                float* dst, int dst_width, int dst_height, const YoloInitParam& param,
                cudaStream_t stream)
{
    int total             = batch_size * src_width * src_height * 3;
    int threads_per_block = 256;
    int blocks            = (total + threads_per_block - 1) / threads_per_block;
    normKernel<<<blocks, threads_per_block, 0, stream>>>(batch_size, src, src_width, src_height,
                                                         dst, dst_width, dst_height,
                                                         param.mean_, param.stds_, param.scale_);
}

// HWC 转 CHW 核函数
__global__ void hwc2ChwKernel(int batch_size, float* src, int src_width, int src_height,
                              float* dst, int dst_width, int dst_height)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * src_width * src_height * 3;
    if (idx >= total)
        return;

    int channel = idx % 3;                              // 通道
    int src_y   = (idx / 3) % src_height;               // 高度
    int src_x   = (idx / (3 * src_height)) % src_width; // 宽度
    int batch   = idx / (3 * src_width * src_height);   // 批次

    int hwc_idx  = (batch * src_height * src_width + src_y * src_width + src_x) * 3 + channel;
    int chw_idx  = (batch * 3 * dst_height * dst_width) + (channel * dst_height * dst_width) + (src_y * dst_width) + src_x;
    dst[chw_idx] = src[hwc_idx];
}

void hwc2ChwDevice(int batch_size, float* src, int src_width, int src_height,
                   float* dst, int dst_width, int dst_height, cudaStream_t stream)
{
    int total             = batch_size * src_width * src_height * 3;
    int threads_per_block = 256;
    int blocks            = (total + threads_per_block - 1) / threads_per_block;
    hwc2ChwKernel<<<blocks, threads_per_block, 0, stream>>>(batch_size, src, src_width, src_height,
                                                            dst, dst_width, dst_height);
}