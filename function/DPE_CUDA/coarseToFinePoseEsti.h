#pragma once

#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <vector_types.h>
#include "apeCommon.h"

namespace ape {

void coarseToFinePoseEstimation(const cv::cuda::GpuMat &tmp,
                                const cv::cuda::GpuMat &img,
                                int prm_lvls,
                                bool photo_inva,
                                bool verbose,
                                ApeParams *ape_params,
                                double *ex_mat);

void rescale(const cv::cuda::GpuMat &tmp,
             const cv::cuda::GpuMat &img,
             const ApeParams &ape_params,
             const float scale,
             cv::cuda::GpuMat *small_tmp,
             cv::cuda::GpuMat *small_img,
             ApeParams *small_ape_params);

float getTotalVariation(const cv::cuda::GpuMat &tmp,
                        int tw,
                        int th,
                        float area);

__global__
void variationKernel(const cv::cuda::PtrStepSz<float3> tmp,
                     int2 dim,
                     float* variation);

void randSample(const cv::cuda::GpuMat &tmp,
                int2 tmp_dim,
                float2 tmp_real,
                int sample_num,
                thrust::device_vector<float2> *tmp_coors,
                thrust::device_vector<float3> *tmp_vals);

__global__
void randSampleKernel(const int2 *rand_coor,
                      const cv::cuda::PtrStepSz<float3> tmp,
                      int2 tmp_dim,
                      float2 tmp_real,
                      int sample_num,
                      float2 *tmp_coors,
                      float3 *tmp_vals);

void createSet(const ApeParams &ape_params,
               thrust::device_vector<Pose> *poses,
               size_t *num_poses);

__global__
void createSetKernel(int begin_index,
                     int4 count,
                     int num,
                     float tz,
                     float rx,
                     ApeParams ape_params,
                     float2 bound,
                     float length,
                     int area_thres,
                     Pose *poses,
                     bool* valids);

void calDist(const thrust::device_vector<Pose> &poses,
             float4 in_params,
             float2 tmp_real,
             int2 img_dim,
             bool photo_inva,
             size_t num_poses,
             size_t sample_num,
             thrust::device_vector<float> *dists);

__global__
void getTmpY(float *tmp_y,
             int sample_num);

__global__
void calDistInvarKernel(const Pose *poses,
                        float4 in_params,
                        float2 tmp_real,
                        int2 img_dim,
                        size_t num_poses,
                        size_t sample_num,
                        float inv_sample_num,
                        float mean_tmp,
                        float sig_tmp,
                        float *dists);

__global__
void calDistColorKernel(const Pose *poses,
                        float4 in_params,
                        float2 tmp_real,
                        int2 img_dim,
                        size_t num_poses,
                        size_t sample_num,
                        float inv_sample_num,
                        float *dists);

float calLastThreeTermsMean(thrust::host_vector<float> &min_dists,
                            int iter_times);

void getExMat(const Pose &poses,
              double *ex_mat);

bool getPosesByDistance(const thrust::device_vector<float> &dists,
                        float min_dist,
                        float epsilon,
                        thrust::device_vector<Pose> *poses,
                        size_t *num_poses);

__global__
void getPosesByDistanceKernel(const float *dists,
                              float threshold,
                              size_t num_poses,
                              bool *survivals);

void expandPoses(float factor,
                 thrust::device_vector<Pose> *poses,
                 ApeParams *ape_params,
                 size_t *num_poses);

__global__
void expandPosesKernel(size_t num_poses,
                       size_t new_num_poses,
                       ApeParams ape_params,
                       int area_thres,
                       Pose *poses,
                       bool *valids);

__global__
void fetchTzKernel(const Pose *poses,
                   size_t num_poses,
                   float *tzs);

}  // namespace ape
                   