#ifndef __COMMON_COMMON_TYPE__
#define __COMMON_COMMON_TYPE__

#include <iostream>
#include <vector>
#include <cfloat>
#include <cstring>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

struct YoloInitParam
{
    bool                     dynamic_batch_{true}; // 是否动态batch
    bool                     is_save_{false};
    bool                     is_show_{false};
    int                      num_class_{80}; // 类型数量，默认COCO数据集为80
    int                      src_w_;
    int                      src_h_;
    int                      dst_w_;
    int                      dst_h_;
    int                      top_k_{300}; // nms后保留的框数量
    int                      char_width_{11};
    int                      det_info_render_width_{15}; // 检测信息渲染宽度
    float                    scale_{255.0F};             // 图像归一化时的scale
    float                    mean_[3]{0.0F, 0.0F, 0.0F}; // 图像归一化时的mean
    float                    stds_[3]{1.0F, 1.0F, 1.0F}; // 图像归一化时的std
    float                    iou_thresh_;                // nms阈值
    float                    conf_thresh_;               // 置信度阈值
    double                   font_scale_{0.6};           // 字体缩放比例
    std::string              save_path_;
    std::string              winname_{"yolo"};
    size_t                   batch_size_; // batch_size
    std::vector<std::string> class_names_;
    std::vector<std::string> input_output_names_; // 输入输出层张量名称
};

struct CandidateObject // 候选框
{
    CandidateObject()
    {
        std::fill_n(bbox_and_key_points_, 14, FLT_MAX); // 将bbox_and_key_points_数组中的元素设置为FLT_MAX
        score_   = FLT_MAX;
        is_good_ = true;
    }

    CandidateObject(float* bbox_and_key_points, float score, bool is_good)
        : score_(score)
        , is_good_(is_good)
    {
        memcpy(bbox_and_key_points_, bbox_and_key_points, 14 * sizeof(float));
    }

    float bbox_and_key_points_[14]; // 边界框（x, y, w, h）+ 5个面部关键点（x1, y1, ..., x5, y5）。
    float score_;                   // 置信度
    bool  is_good_;                 // 是否有效
};

struct Box
{
    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label)
        : left_(left)
        , top_(top)
        , right_(right)
        , bottom_(bottom)
        , confidence_(confidence)
        , label_(label)
    {
    }

    Box(float left, float top, float right, float bottom, float confidence, int label, int land_marks_num)
        : left_(left)
        , top_(top)
        , right_(right)
        , bottom_(bottom)
        , confidence_(confidence)
        , label_(label)
    {
        land_marks_.reserve(land_marks_num);
    }

    float                    left_;
    float                    top_;
    float                    right_;
    float                    bottom_;
    float                    confidence_;
    int                      label_;
    std::vector<cv::Point2i> land_marks_; // 关键点坐标
};

enum class InputStream
{
    IMAGE,
    VIDEO,
    CAMERA
};

enum class ColorMode
{
    RGB,
    GRAY
};

struct AffineMat // 仿射变换矩阵
{
    float v0_;
    float v1_;
    float v2_;
    float v3_;
    float v4_;
    float v5_;
};

class HostTimer
{
public:
    HostTimer()
        : t1(std::chrono::steady_clock::now())
    {
    }

    float getUsedTime() const
    {
        auto t2 = std::chrono::steady_clock::now();
        return (std::chrono::duration<float, std::milli>(t2 - t1).count()); // 直接转换为毫秒
    }

private:
    std::chrono::steady_clock::time_point t1;
};

class DeviceTimer
{
public:
    DeviceTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }

    DeviceTimer(cudaStream_t stream)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, stream);
    }

    ~DeviceTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float getUsedTime()
    {
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float total_time;
        cudaEventElapsedTime(&total_time, start, end);
        return total_time;
    }

    float getUsedTime(cudaStream_t stream)
    {
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        float total_time;
        cudaEventElapsedTime(&total_time, start, end);
        return total_time;
    }

private:
    cudaEvent_t start;
    cudaEvent_t end;
};

#endif // __COMMON_COMMON_TYPE__