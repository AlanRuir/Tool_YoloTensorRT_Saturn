#include "yolo_base.h"
#include "kernel_function.h"

YoloBase::YoloBase(const YoloInitParam& param)
    : param_(param)
    , engine_(nullptr)
    , context_(nullptr)
    , input_src_device_(nullptr)
    , input_resize_device_(nullptr)
    , input_rgb_device_(nullptr)
    , input_norm_device_(nullptr)
    , input_hwc_device_(nullptr)
    , output_src_device_(nullptr)
    , output_objects_device_(nullptr)
    , output_objects_host_(nullptr)
    , output_conf_device_(nullptr)
    , output_idx_device_(nullptr)
    , output_objects_width_(7)
{
    CHECK(cudaMalloc(&input_src_device_, param_.batch_size_ * param_.src_w_ * param_.src_h_ * CHANNEL_NUM * sizeof(uint8_t)));
    CHECK(cudaMalloc(&input_resize_device_, param_.batch_size_ * param_.dst_w_ * param_.dst_h_ * CHANNEL_NUM * sizeof(float)));
    CHECK(cudaMalloc(&input_rgb_device_, param_.batch_size_ * param_.dst_w_ * param_.dst_h_ * CHANNEL_NUM * sizeof(float)));
    CHECK(cudaMalloc(&input_norm_device_, param_.batch_size_ * param_.dst_w_ * param_.dst_h_ * CHANNEL_NUM * sizeof(float)));
    CHECK(cudaMalloc(&input_hwc_device_, param_.batch_size_ * param_.dst_w_ * param_.dst_h_ * CHANNEL_NUM * sizeof(float)));

    int output_objects_size = param_.batch_size_ * (1 + param_.top_k_ * output_objects_width_);

    CHECK(cudaMalloc(&output_objects_device_, output_objects_size * sizeof(float)));
    CHECK(cudaMalloc(&output_idx_device_, param_.batch_size_ * param_.top_k_ * sizeof(int)));
    CHECK(cudaMalloc(&output_conf_device_, param_.batch_size_ * param_.top_k_ * sizeof(float)));

    output_objects_host_ = new float[output_objects_size];
    objectss_.resize(param_.batch_size_);
}

YoloBase::~YoloBase()
{
    CHECK(cudaFree(input_src_device_));
    CHECK(cudaFree(input_resize_device_));
    CHECK(cudaFree(input_rgb_device_));
    CHECK(cudaFree(input_norm_device_));
    CHECK(cudaFree(input_hwc_device_));

    CHECK(cudaFree(output_objects_device_));
    CHECK(cudaFree(output_idx_device_));
    CHECK(cudaFree(output_conf_device_));

    delete[] output_objects_host_;

    if (stream_)
    {
        cudaStreamDestroy(stream_);
    }
}

bool YoloBase::init(const std::vector<uint8_t>& trt_engine_file)
{
    if (trt_engine_file.empty())
    {
        std::cerr << "Failed to load engine file" << std::endl;
        return false;
    }

    /* 创建推理引擎 */
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(global_logger));
    if (!runtime)
    {
        std::cerr << "Failed to create infer runtime" << std::endl;
        return false;
    }

    /* 反序列化引擎 */
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trt_engine_file.data(), trt_engine_file.size()),
        [](nvinfer1::ICudaEngine* engine) {
            if (engine)
            {
                delete engine; // 注意：TensorRT 10.x 不需要手动 delete，这里仅为兼容旧习惯
            }
        });

    if (!engine_)
    {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    /* 创建推理上下文 */
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    /* 创建CUDA流 */
    cudaError_t status = cudaStreamCreate(&stream_);
    if (status != cudaSuccess)
    {
        std::cerr << "Failed to create cuda stream" << std::endl;
        return false;
    }

    /* 配置动态批次 */
    if (param_.dynamic_batch_)
    {
        // 使用张量名称而不是索引
        if (!context_->setInputShape(param_.input_output_names_[0].c_str(),
                                     nvinfer1::Dims4(param_.batch_size_, CHANNEL_NUM, param_.src_h_, param_.src_w_)))
        {
            std::cerr << "Failed to set input shape for " << param_.input_output_names_[0] << std::endl;
            cudaStreamDestroy(stream_);
            return false;
        }
    }

    // 获取输出张量形状
    output_dims_ = context_->getTensorShape(param_.input_output_names_[1].c_str());
    if (output_dims_.nbDims == -1) // 检查是否有效
    {
        std::cerr << "Failed to get output tensor shape for " << param_.input_output_names_[1] << std::endl;
        cudaStreamDestroy(stream_);
        return false;
    }

    total_objects_ = output_dims_.d[1];
    if (param_.batch_size_ > static_cast<size_t>(output_dims_.d[0])) // 统一类型
    {
        std::cerr << "Batch size is too large" << std::endl;
        cudaStreamDestroy(stream_);
        return false;
    }

    /* 计算输出区域并分配内存 */
    output_area_ = 1;
    for (int i = 0; i < output_dims_.nbDims; ++i)
    {
        if (output_dims_.d[i] != 0)
        {
            output_area_ *= output_dims_.d[i];
        }
    }

    status = cudaMalloc(&output_src_device_, param_.batch_size_ * output_area_ * sizeof(float));
    if (status != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory" << std::endl;
        cudaStreamDestroy(stream_);
        return false;
    }

    /* 计算仿射变换矩阵 */
    float   a       = static_cast<float>(param_.dst_h_) / param_.src_h_;
    float   b       = static_cast<float>(param_.dst_w_) / param_.src_w_;
    float   scale   = std::min(a, b);
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * param_.src_w_ + param_.dst_w_ + scale - 1) * 0.5,
                       0.f, scale, (-scale * param_.src_h_ + param_.dst_h_ + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);
    dst2src_.v0_ = dst2src.ptr<float>(0)[0];
    dst2src_.v1_ = dst2src.ptr<float>(0)[1];
    dst2src_.v2_ = dst2src.ptr<float>(0)[2];
    dst2src_.v3_ = dst2src.ptr<float>(1)[0];
    dst2src_.v4_ = dst2src.ptr<float>(1)[1];
    dst2src_.v5_ = dst2src.ptr<float>(1)[2];

    std::cout << "YoloBase init success" << std::endl;
    return true;
}

void YoloBase::check()
{
    std::cout << "[INFO] the engine's info:" << std::endl;
    for (const auto& layer_name : param_.input_output_names_)
    {
        nvinfer1::Dims dims = engine_->getTensorShape(layer_name.c_str());
        std::cout << "[INFO] " << layer_name << ": ";
        if (dims.nbDims == -1)
        {
            std::cout << "Invalid tensor shape" << std::endl;
        }
        else
        {
            for (int i = 0; i < dims.nbDims; i++)
            {
                std::cout << dims.d[i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "[INFO] the context's info:" << std::endl;
    for (const auto& layer_name : param_.input_output_names_)
    {
        nvinfer1::Dims dims = context_->getTensorShape(layer_name.c_str());
        std::cout << "[INFO] " << layer_name << ": ";
        if (dims.nbDims == -1)
        {
            std::cout << "Invalid tensor shape" << std::endl;
        }
        else
        {
            for (int i = 0; i < dims.nbDims; i++)
            {
                std::cout << dims.d[i] << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void YoloBase::copy(const std::vector<cv::Mat>& imgs_batch)
{
    unsigned char* pi = input_src_device_;
    for (size_t i = 0; i < imgs_batch.size(); i++)
    {
        cudaError_t status = cudaMemcpyAsync(
            pi, imgs_batch[i].data,
            sizeof(unsigned char) * 3 * param_.src_h_ * param_.src_w_,
            cudaMemcpyHostToDevice,
            stream_);
        if (status != cudaSuccess)
        {
            std::cout << "[ERROR] cudaMemcpyAsync failed for image " << i << ": "
                      << cudaGetErrorString(status) << std::endl;
            return;
        }
        pi += sizeof(unsigned char) * 3 * param_.src_h_ * param_.src_w_;
    }
}

void YoloBase::preprocess(const std::vector<cv::Mat>& imgs_batch)
{
    /* 下述操作需要核函数支持流操作 */
    resizeDevice(param_.batch_size_, input_src_device_, param_.src_w_, param_.src_h_,
                 input_resize_device_, param_.dst_w_, param_.dst_h_, 114, dst2src_, stream_);
    bgr2RgbDevice(param_.batch_size_, input_resize_device_, param_.dst_w_, param_.dst_h_,
                  input_rgb_device_, param_.dst_w_, param_.dst_h_, stream_);
    normDevice(param_.batch_size_, input_rgb_device_, param_.dst_w_, param_.dst_h_,
               input_norm_device_, param_.dst_w_, param_.dst_h_, param_, stream_);
    hwc2ChwDevice(param_.batch_size_, input_norm_device_, param_.dst_w_, param_.dst_h_,
                  input_hwc_device_, param_.dst_w_, param_.dst_h_, stream_);
}

bool YoloBase::infer()
{
    // 设置输入和输出张量的地址
    if (!context_->setInputTensorAddress(param_.input_output_names_[0].c_str(), input_hwc_device_) || !context_->setOutputTensorAddress(param_.input_output_names_[1].c_str(), output_src_device_))
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        return false;
    }

    // 执行推理
    bool enqueue_status = context_->enqueueV3(stream_);
    if (!enqueue_status)
    {
        std::cerr << "Failed to infer" << std::endl;
        return false;
    }

    // 同步流
    cudaError_t sync_status = cudaStreamSynchronize(stream_);
    if (sync_status != cudaSuccess)
    {
        std::cerr << "Failed to synchronize stream: " << cudaGetErrorString(sync_status) << std::endl;
        return false;
    }

    return true;
}

void YoloBase::postprocess(const std::vector<cv::Mat>& imgs_batch) // 有函数未实现
{
    //     decodeDevice(param_, output_src_device_, 5 + param_.num_class_, total_objects_, output_area_,
    //                  output_objects_device_, output_objects_width_, param_.top_k_, stream_);
    //     nmsDeviceV1(param_, output_objects_device_, output_objects_width_, param_.top_k_,
    //                 param_.top_k_ * output_objects_width_ + 1, stream_);

    //     cudaError_t status = cudaMemcpyAsync(
    //         output_objects_host_, output_objects_device_,
    //         param_.batch_size_ * sizeof(float) * (1 + 7 * param_.top_k_),
    //         cudaMemcpyDeviceToHost, stream_);
    //     if (status != cudaSuccess)
    //     {
    //         std::cout << "[ERROR] cudaMemcpyAsync failed: " << cudaGetErrorString(status) << std::endl;
    //         return;
    //     }

    //     // 同步流，确保数据拷贝完成
    //     status = cudaStreamSynchronize(stream_);
    //     if (status != cudaSuccess)
    //     {
    //         std::cout << "[ERROR] cudaStreamSynchronize failed: " << cudaGetErrorString(status) << std::endl;
    //         return;
    //     }

    //     for (size_t bi = 0; bi < imgs_batch.size(); bi++)
    //     {
    //         int num_boxes = std::min((int)(output_objects_host_ + bi * (param_.top_k_ * output_objects_width_ + 1))[0], param_.top_k_);
    //         for (size_t i = 0; i < num_boxes; i++)
    //         {
    //             float* ptr       = output_objects_host_ + bi * (param_.top_k_ * output_objects_width_ + 1) + output_objects_width_ * i + 1;
    //             int    keep_flag = ptr[6];
    //             if (keep_flag)
    //             {
    //                 float x_lt = dst2src_.v0_ * ptr[0] + dst2src_.v1_ * ptr[1] + dst2src_.v2_;
    //                 float y_lt = dst2src_.v3_ * ptr[0] + dst2src_.v4_ * ptr[1] + dst2src_.v5_;
    //                 float x_rb = dst2src_.v0_ * ptr[2] + dst2src_.v1_ * ptr[3] + dst2src_.v2_;
    //                 float y_rb = dst2src_.v3_ * ptr[2] + dst2src_.v4_ * ptr[3] + dst2src_.v5_;
    //                 objectss_[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
    //             }
    //         }
    //     }
}

void YoloBase::reset()
{
    cudaError_t status = cudaMemsetAsync(
        output_objects_device_, 0,
        sizeof(float) * param_.batch_size_ * (1 + 7 * param_.top_k_),
        stream_);
    if (status != cudaSuccess)
    {
        std::cout << "[ERROR] cudaMemsetAsync failed: " << cudaGetErrorString(status) << std::endl;
        return;
    }

    for (size_t bi = 0; bi < param_.batch_size_; bi++)
    {
        objectss_[bi].clear();
    }
}