#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "utils.h"

#define CHECK(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// 缩放图像
void resizeDevice(int batch_size, unsigned char* src, int src_width, int src_height,
                  float* dst, int dst_width, int dst_height, float padding_value,
                  AffineMat dst2src, cudaStream_t stream = 0);

// BGR 转 RGB
void bgr2RgbDevice(int batch_size, float* src, int src_width, int src_height,
                   float* dst, int dst_width, int dst_height, cudaStream_t stream = 0);

// 归一化
void normDevice(int batch_size, float* src, int src_width, int src_height,
                float* dst, int dst_width, int dst_height, const YoloInitParam& param,
                cudaStream_t stream = 0);

// HWC 转 CHW
void hwc2ChwDevice(int batch_size, float* src, int src_width, int src_height,
                   float* dst, int dst_width, int dst_height, cudaStream_t stream = 0);