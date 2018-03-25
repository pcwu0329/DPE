#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "prCommon.h"

namespace pr {

struct VarianceFunctor : std::unary_function<float, float>
{
    float mean;

    VarianceFunctor(float _mean) : mean(_mean) {}

    __host__ __device__
    float operator()(float value) const {
        return ::powf(value - mean, 2.0f);
    }
};

void preCal(const cv::Mat &tmp,
            const cv::Mat &img,
            const std::vector<cv::Mat> &ex_mats,
            float fx,
            float fy,
            float cx,
            float cy,
            float min_dim,
            int prm_lvls,
            bool photo_inva,
            cv::cuda::GpuMat *tmp_d,
            cv::cuda::GpuMat *img_d,
            RpParams *rp_params);

__global__
void getTmpYKernel(const cv::cuda::PtrStepSz<float3> tmp,
                   const int tmp_size,
                   const int tw,
                   float *tmp_y);

__global__
void getImgYKernel(const cv::cuda::PtrStepSz<float3> img,
                   const int tmp_size,
                   const int tw,
                   const float4 homo_1st,
                   const float4 homo_2nd,
                   const int iw,
                   const int ih,
                   float *img_y);

__global__
void normalizeImgYKernel(cv::cuda::PtrStepSz<float3> img,
                         const int img_size, 
                         const int iw,
                         const float alpha,
                         const float beta);

}  // namespace pr
