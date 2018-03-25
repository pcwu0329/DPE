#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

namespace pr {

cv::Mat poseRefinement(const cv::Mat &tmp,
                       const cv::Mat &img,
                       const std::vector<cv::Mat> &ex_mats,
                       float fx,
                       float fy,
                       float cx,
                       float cy,
                       float min_dim,
                       int prm_lvls,
                       bool photo_inva,
                       bool verbose);

cv::Mat loadMatrix(std::string fileName);

}  // namespace pr
