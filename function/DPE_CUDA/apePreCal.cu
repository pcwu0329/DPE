#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <algorithm>
#include <mex.h>
#include "approxPoseEsti.h"
#include "apePreCal.h"
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
            ApeParams *ape_params) {
    ape_params->tw = tmp.cols;
    ape_params->th = tmp.rows;
    ape_params->iw = img.cols;
    ape_params->ih = img.rows;

    // intrinsic parameter
    ape_params->fx = fx;
    ape_params->fy = fy;
    ape_params->cx = cx;
    ape_params->cy = cy;

    // search range in pose domain
    float inv_min_tmp_side_length = 1.f / fminf(ape_params->th, ape_params->tw);
    ape_params->tmp_real_w = inv_min_tmp_side_length * min_dim * ape_params->tw;
    ape_params->tmp_real_h = inv_min_tmp_side_length * min_dim * ape_params->th;
    
    ape_params->min_rz = float(-M_PI);
    ape_params->max_rz = float(M_PI);
    ape_params->min_rx = 0.f;
    ape_params->max_rx = 80 * float(M_PI) / 180;
    ape_params->min_tz = min_tz;
    ape_params->max_tz = max_tz;

    // bounds
    float m_tz = sqrt(min_tz*max_tz);
    float sqrt2 = sqrt(2.f);
    float invtmp = 1.f / sqrt2 / m_tz;
    ape_params->epsilon = epsilon;
    ape_params->step.rz0 = epsilon * sqrt2;
    ape_params->step.rx = epsilon * invtmp;
    ape_params->step.rz1 = epsilon * sqrt2;
    ape_params->step.tx = epsilon * invtmp * 2 * (min_dim);
    ape_params->step.ty = epsilon * invtmp * 2 * (min_dim);
    ape_params->step.tz = epsilon * invtmp;

    // allocate memory to GPU
    cv::cuda::GpuMat tmp_ori(tmp);
    cv::cuda::GpuMat img_ori(img);
    cv::cuda::GpuMat tmp_nmlz(ape_params->th, ape_params->tw, CV_32FC3);
    cv::cuda::GpuMat img_nmlz(ape_params->ih, ape_params->iw, CV_32FC3);
    tmp_ori.convertTo(tmp_nmlz, CV_32FC3, 1.0 / 255.0);
    img_ori.convertTo(img_nmlz, CV_32FC3, 1.0 / 255.0); 
    
    // transform RGB to YCbCr
    cv::cuda::cvtColor(tmp_nmlz, *tmp_d, CV_BGR2YCrCb);
    cv::cuda::cvtColor(img_nmlz, img_nmlz, CV_BGR2YCrCb);
    cv::cuda::cvtColor(img_nmlz, *img_d, CV_BGR2BGRA, 4);
}

}  // namespace ape
