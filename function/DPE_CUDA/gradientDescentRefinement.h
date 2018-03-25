#pragma once

#include <opencv2/core/cuda_types.hpp>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <vector_types.h>
#include "prCommon.h"

namespace pr {

cv::Mat GradientDescentRefinement(const cv::cuda::GpuMat &tmp,
                                  const cv::cuda::GpuMat &img,
                                  const RpParams &rp_params,
                                  const std::vector<cv::Mat> &ex_mats,
                                  bool verbose);

bool CalValidCoors(const cv::cuda::GpuMat &tmp,
                   const cv::Mat &in_mat,
                   const cv::Mat &ex_mat,
                   const cv::Mat &nm_mat,
                   int tw,
                   int th,
                   int iw,
                   int ih,
                   thrust::device_vector<float4> *tmp_vals,
                   thrust::device_vector<float2> *tmp_coors);

__global__
void CalValidCoorsKernel(int4 rect,
                         const int num,
                         const int2 tmp_size,
                         const float4 nm_params,
                         const float4 homo_1st,
                         const float4 homo_2nd,
                         float4 *tmp_vals,
                         float2 *tmp_coors,
                         bool *valids);

// Return appearance error
float Gdr(const thrust::device_vector<float4> &tmp_vals,
          const thrust::device_vector<float2> &tmp_coors,     
          const cv::Mat &in_mat,
          int2 img_size,
          float2 tmp_real,
          float epsilon_r,
          float epsilon_t,
          int max_iter,
          float scale,
          bool verbose,
          cv::Mat &ex_mat);

void CalAprErrAndGradient(const thrust::device_vector<float4> &tmp_vals,
                          const thrust::device_vector<float2> &tmp_coors,
                          int num,
                          const cv::Mat &in_mat,
                          const cv::Mat &ex_mat,
                          int2 img_size,
                          float *apr_errs,
                          thrust::device_vector<float4> *img_u_vals,
                          thrust::device_vector<float4> *img_v_vals);

__global__
void CalAprErrAndGradientKernel(const float4 *tmp_vals,
                                const float2 *tmp_coors,
                                const int num,
                                const int2 img_size,
                                const float4 homo_1st,
                                const float4 homo_2nd,
                                float *apr_errs,
                                float4 *img_u_vals,
                                float4 *img_v_vals);

bool WithinRegion(int2 img_size, 
                  float2 tmp_real,
                  float4 homo_1st,
                  float4 homo_2nd);

__global__
void CalJacobianKernel(const float2 *tmp_coors,
                       const float4 *img_u_vals,
                       const float4 *img_v_vals,
                       const int num,
                       const float2 in_mat_params,
                       const R12t R12_t,
                       const JacobRr J_Rr,
                       float *J);

}  // namespace pr
