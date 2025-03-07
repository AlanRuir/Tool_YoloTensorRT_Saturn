#ifndef __YOLO_YOLO_BASE__
#define __YOLO_YOLO_BASE__

#include <iostream>
#include <memory>
// #include <parserOnnxConfig.h>
#include <NvInfer.h>
#include <vector>

#include "common_type.hpp"
#include "data_set.hpp"
#include "utils.h"
#include "simple_logger.hpp"

class YoloBase
{
public:
    YoloBase(const YoloInitParam& param);
    ~YoloBase();

    virtual bool init(const std::vector<uint8_t>& trt_engine_file);
    virtual void check();
    virtual void copy(const std::vector<cv::Mat>& imgs_batch);
    virtual void preprocess(const std::vector<cv::Mat>& imgs_batch);
    virtual bool infer(); // 异步推理
    virtual void postprocess(const std::vector<cv::Mat>& imgs_batch);
    virtual void reset();

    std::vector<std::vector<Box>> getObjectss() const;

protected:
    YoloInitParam                                param_;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;      // 推理引擎
    std::unique_ptr<nvinfer1::IExecutionContext> context_;     // 推理上下文
    nvinfer1::Dims                               output_dims_; // 张量维度
    int                                          output_area_; // 张量面积
    int                                          total_objects_;
    std::vector<std::vector<Box>>                objectss_;
    AffineMat                                    dst2src_; // 仿射变换矩阵
    uint8_t*                                     input_src_device_;
    float*                                       input_resize_device_;
    float*                                       input_rgb_device_;
    float*                                       input_norm_device_;
    float*                                       input_hwc_device_;
    float*                                       output_src_device_;
    float*                                       output_objects_device_;
    float*                                       output_objects_host_;
    float*                                       output_conf_device_;
    int*                                         output_idx_device_;
    int                                          output_objects_width_; // 对象的属性数量
    cudaStream_t                                 stream_;               // 推理流

private:
    static constexpr int CHANNEL_NUM = 3;
};

#endif // __YOLO_YOLO_BASE__