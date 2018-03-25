#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

namespace ape {

void approxPoseEsti(const cv::Mat &tmp,
                    const cv::Mat &img,
                    float fx,
                    float fy,
                    float cx,
                    float cy,
                    float min_dim,
                    float min_tz,
                    float max_tz,
                    float epsilon,
                    int prm_lvls,
                    bool photo_inva,
                    bool verbose,
                    double *ex_mat);

}  // namespace ape