#include <fstream>

#include "utils.h"
#include "data_set.hpp"

void saveBinaryFile(float* vec, size_t len, const std::string& file)
{
    std::ofstream out(file, std::ios::out | std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Failed to open file: " + file);
    }
    out.write(reinterpret_cast<const char*>(vec), sizeof(float) * len);
    if (!out)
    {
        throw std::runtime_error("Failed to write data to file: " + file);
    }
    out.close();
}

std::vector<uint8_t> readBinaryFile(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + file);
    }
    in.seekg(0, std::ios::end);
    size_t len = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(len);

    in.read(reinterpret_cast<char*>(buffer.data()), len);
    if (!in)
    {
        throw std::runtime_error("Failed to read data from file: " + file);
    }

    in.close();
    return buffer;
}

std::vector<uint8_t> loadModel(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + file);
    }
    in.seekg(0, std::ios::end);
    size_t len = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(len);

    in.read(reinterpret_cast<char*>(buffer.data()), len);
    if (!in)
    {
        throw std::runtime_error("Failed to read data from file: " + file);
    }

    in.close();
    return buffer;
}

std::string getSystemTimeStr()
{
    auto        now       = std::chrono::system_clock::now();
    auto        in_time_t = std::chrono::system_clock::to_time_t(now);
    std::string str       = std::to_string(in_time_t);
    return str;
}

bool setInputStream(const InputStream& source, const std::string& image_path, const std::string& video_path, const int& camera_ID,
                    cv::VideoCapture& capture, int& total_batches, int& delay_time, YoloInitParam& param)
{
    // 后续补全
}

void setRenderWindow(YoloInitParam& param)
{
    if (!param.is_show_)
    {
        return;
    }

    int   max_width    = 960;
    int   max_height   = 540;
    float scale_width  = static_cast<float>(param.src_w_ / max_width);
    float scale_height = static_cast<float>(param.src_h_ / max_height);

    if (scale_width > 1.0F && scale_height > 1.0F)
    {
        float scale = std::min(scale_width, scale_height);
        cv::namedWindow(param.winname_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO); // cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO的意思是可以自动调整窗口大小、
        cv::resizeWindow(param.winname_, static_cast<int>(param.src_w_ / scale), static_cast<int>(param.src_h_ / scale));
        param.char_width_            = 16;
        param.det_info_render_width_ = 18;
        param.font_scale_            = 0.9;
    }
    else
    {
        cv::namedWindow(param.winname_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        param.char_width_            = 11;
        param.det_info_render_width_ = 15;
        param.font_scale_            = 0.6;
    }
}

std::string getTimeStamp()
{
    std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    return std::to_string(ns.count());
}

void show(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& class_names,
          const int& cv_delay_time, std::vector<cv::Mat>& imgs_batch)
{
    if (imgs_batch[0].empty())
    {
        return;
    }

    std::string windows_title = "image";

    cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO); // allow window resize(Linux)

    int max_width  = 960;
    int max_height = 540;
    if (imgs_batch[0].rows > max_height || imgs_batch[0].cols > max_width)
    {
        cv::resizeWindow(windows_title, max_width, imgs_batch[0].rows * max_width / imgs_batch[0].cols);
    }

    cv::Scalar       color = cv::Scalar(0, 255, 0); // BGR
    cv::Point        bbox_points[1][4];             // 4个点
    const cv::Point* bbox_point0[1]{bbox_points[0]};
    int              points_nums[1]{4};

    for (size_t i = 0; i < imgs_batch.size(); ++i)
    {
        if (!objectss.empty())
        {
            for (auto& box : objectss[i])
            {
                if (80 == class_names.size())
                {
                    color = color80[box.label_];
                }
                else if (91 == class_names.size())
                {
                    color = color91[box.label_];
                }
                else if (20 == class_names.size())
                {
                    color = color20[box.label_];
                }

                cv::rectangle(imgs_batch[i], cv::Point(box.left_, box.top_), cv::Point(box.right_, box.bottom_), color, 2, cv::LINE_AA);
                cv::String det_info = class_names[box.label_] + " " + cv::format("%.4f", box.confidence_);

                bbox_points[0][0] = cv::Point(box.left_, box.top_);
                bbox_points[0][1] = cv::Point(box.left_ + det_info.size() * 11, box.top_);
                bbox_points[0][2] = cv::Point(box.left_ + det_info.size() * 11, box.top_ - 15);
                bbox_points[0][3] = cv::Point(box.left_, box.top_ - 15);

                cv::fillPoly(imgs_batch[i], bbox_point0, points_nums, 1, color);
                cv::putText(imgs_batch[i], det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                if (!box.land_marks_.empty()) // for facial landmarks
                {
                    for (auto& x : box.land_marks_)
                    {
                        cv::circle(imgs_batch[i], x, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
                    }
                }
            }
        }

        cv::imshow(windows_title, imgs_batch[i]);
        cv::waitKey(cv_delay_time);
    }
}