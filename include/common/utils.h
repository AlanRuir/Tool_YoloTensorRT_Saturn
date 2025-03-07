#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <iostream>
#include <vector>

#include "common_type.hpp"

void                 saveBinaryFile(float* vec, size_t len, const std::string& file);
std::vector<uint8_t> readBinaryFile(const std::string& file);
std::vector<uint8_t> loadModel(const std::string& file);
std::string          getSystemTimeStr();
bool                 setInputStream(const InputStream& source, const std::string& image_path, const std::string& video_path, const int& camera_ID,
                                    cv::VideoCapture& capture, int& total_batches, int& delay_time, YoloInitParam& param);
void                 setRenderWindow(YoloInitParam& param);
std::string          getTimeStamp();
void                 show(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& class_names,
                          const int& cv_delay_time, std::vector<cv::Mat>& imgs_batch);
void                 save(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& class_names,
                          const std::string& save_path, std::vector<cv::Mat>& imgs_batch, const int& batch_size, const int& batchi);
#endif // __COMMON_UTILS__