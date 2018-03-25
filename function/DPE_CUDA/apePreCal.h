#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "apeCommon.h"

namespace ape {

void preCal(const cv::Mat &tmp,
            const cv::Mat &img,
            float fx,
            float fy,
            float cx,
            float cy,
            float min_dim,
            float min_tz,
            float max_tz,
            float epsilon,
            bool verbose, 
            cv::cuda::GpuMat *tmp_d,
            cv::cuda::GpuMat *img_d,
            ApeParams *ape_params);

}  // namespace ape
