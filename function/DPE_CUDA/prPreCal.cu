#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include "poseRefinement.h"
#include "prPreCal.h"
#include "prCommon.h"
#include "prUtil.h"

namespace pr {

static const int BLOCK_SIZE_1D = 256;

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
            RpParams *rp_params) {
    Timer time;
    time.Reset();
    time.Start();

    rp_params->prm_lvls = prm_lvls;

    size_t pose_num = ex_mats.size();
    rp_params->tw = tmp.cols;
    rp_params->th = tmp.rows;
    rp_params->iw = img.cols;
    rp_params->ih = img.rows;

    // intrinsic matrix
    rp_params->in_mat = (cv::Mat_<float>(3, 3) << fx,  0, cx,
                                                   0, fy, cy,
                                                   0,  0,  1);

    // real template size
    float inv_min_tmp_side_length = 1.f / fminf(rp_params->th, rp_params->tw);
    rp_params->tmp_real_w = inv_min_tmp_side_length * min_dim * rp_params->tw;
    rp_params->tmp_real_h = inv_min_tmp_side_length * min_dim * rp_params->th;

    // template normalization matrix
    float x_center = 0.5f*(rp_params->tw - 1);
    float y_center = 0.5f*(rp_params->th - 1);

    // cv::Mat::eye(3, 3, CV_32F) is quite slow (50ms); use the following way instead (1e-3ms)
    rp_params->nm_mat = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    rp_params->nm_mat.at<float>(0, 0) = 2 * rp_params->tmp_real_w / rp_params->tw;
    rp_params->nm_mat.at<float>(0, 2) = -x_center * rp_params->nm_mat.at<float>(0, 0);
    rp_params->nm_mat.at<float>(1, 1) = -2 * rp_params->tmp_real_h / rp_params->th;
    rp_params->nm_mat.at<float>(1, 2) = -y_center * rp_params->nm_mat.at<float>(1, 1);

    // determine the search range (2x side length)
    Region region;
    float scale = 1.5f;
    region.x_max = 0;
    region.y_max = 0;
    region.x_min = rp_params->iw;
    region.y_min = rp_params->ih;
    cv::Mat boundary = (cv::Mat_<float>(3, 4) << 0, 0, rp_params->tw - 1, rp_params->tw - 1,
                                                 0, rp_params->th - 1, rp_params->th - 1, 0,
                                                 1, 1, 1, 1);

    for (int i = 0; i < pose_num; ++i) {
        cv::Mat H = getHomoMatFromInExNm(rp_params->in_mat, ex_mats[0], rp_params->nm_mat);
        cv::Mat temp_mat;
        Region temp_region;
        calRegion(boundary,
                  H,
                  rp_params->tw,
                  rp_params->th,
                  rp_params->iw,
                  rp_params->ih,
                  scale,
                  &temp_mat,
                  &temp_region);
        region.x_max = std::max(region.x_max, temp_region.x_max);
        region.y_max = std::max(region.y_max, temp_region.y_max);
        region.x_min = std::min(region.x_min, temp_region.x_min);
        region.y_min = std::min(region.y_min, temp_region.y_min);
    }
    rp_params->in_mat.at<float>(0, 2) -= region.x_min;
    rp_params->in_mat.at<float>(1, 2) -= region.y_min;
    rp_params->iw = region.x_max - region.x_min + 1;
    rp_params->ih = region.y_max - region.y_min + 1;

    // allocate memory to GPU
    cv::cuda::GpuMat tmp_ori(tmp);
    cv::cuda::GpuMat img_ori(img(cv::Rect(region.x_min, region.y_min, rp_params->iw, rp_params->ih)));
    cv::cuda::GpuMat tmp_nmlz(rp_params->th, rp_params->tw, CV_32FC3);
    cv::cuda::GpuMat img_nmlz(rp_params->ih, rp_params->iw, CV_32FC3);
    tmp_ori.convertTo(tmp_nmlz, CV_32FC3, 1.0 / 255.0);
    img_ori.convertTo(img_nmlz, CV_32FC3, 1.0 / 255.0);

    // transform RGB to YCbCr
    cv::cuda::cvtColor(tmp_nmlz, tmp_nmlz, CV_BGR2YCrCb);
    cv::cuda::cvtColor(img_nmlz, img_nmlz, CV_BGR2YCrCb);

    if (photo_inva) {
        int tmp_size = rp_params->tw * rp_params->th;
        const int BLOCK_NUM = (tmp_size - 1) / BLOCK_SIZE_1D + 1;
        // about template image
        thrust::device_vector<float> tmp_y(tmp_size);
        getTmpYKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(tmp_nmlz,
                                                    tmp_size,
                                                    rp_params->tw,
                                                    thrust::raw_pointer_cast(tmp_y.data()));
        cudaDeviceSynchronize();
        float mean_tmp = thrust::reduce(tmp_y.cbegin(), tmp_y.cend()) / tmp_size;
        float variance_tmp = thrust::transform_reduce(tmp_y.cbegin(),
                                                      tmp_y.cend(),
                                                      VarianceFunctor(mean_tmp),
                                                      0.0f,
                                                      thrust::plus<float>()) / tmp_size;
        float sig_tmp = sqrt(variance_tmp);
        // about camera frame
        std::vector<float> mean_imgs(pose_num);
        std::vector<float> sig_imgs(pose_num);
        thrust::device_vector<float> img_y(tmp_size);
        for (int i = 0; i < pose_num; ++i) {
            auto H = getHomoMatFromInExNm(rp_params->in_mat, ex_mats[i], rp_params->nm_mat);
            float4 homo_1st = make_float4(H.at<float>(0, 0), H.at<float>(1, 0), H.at<float>(2, 0), H.at<float>(0, 1));
            float4 homo_2nd = make_float4(H.at<float>(1, 1), H.at<float>(2, 1), H.at<float>(0, 2), H.at<float>(1, 2));
            getImgYKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(img_nmlz,
                                                        tmp_size,
                                                        rp_params->tw,
                                                        homo_1st,
                                                        homo_2nd,
                                                        rp_params->iw,
                                                        rp_params->ih,
                                                        thrust::raw_pointer_cast(img_y.data()));
            mean_imgs[i] = thrust::reduce(img_y.cbegin(), img_y.cend()) / tmp_size;
            float variance_img = thrust::transform_reduce(img_y.cbegin(),
                                                          img_y.cend(),
                                                          VarianceFunctor(mean_tmp),
                                                          0.0f,
                                                          thrust::plus<float>()) / tmp_size;
            sig_imgs[i] = sqrt(variance_img);
        }
        float mean_img = std::accumulate(mean_imgs.begin(), mean_imgs.end(), 0.0) / pose_num;
        float sig_img = std::accumulate(sig_imgs.begin(), sig_imgs.end(), 0.0) / pose_num;
        float alpha = sig_tmp / sig_img;
        float beta = -mean_img*alpha + mean_tmp;
        int img_size = rp_params->iw * rp_params->ih;
        normalizeImgYKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(img_nmlz,
                                                          img_size,
                                                          rp_params->iw,
                                                          alpha,
                                                          beta);
    }
    cv::cuda::cvtColor(tmp_nmlz, *tmp_d, CV_BGR2BGRA, 4);
    cv::cuda::cvtColor(img_nmlz, *img_d, CV_BGR2BGRA, 4);
}

__global__
void getTmpYKernel(cv::cuda::PtrStepSz<float3> tmp,
                   const int tmp_size,
                   const int tw,
                   float *tmp_y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tmp_size)
        return;

    int y = idx / tw;
    int x = idx % tw;
    tmp_y[idx] = tmp(y, x).x;
}

__global__
void getImgYKernel(const cv::cuda::PtrStepSz<float3> img,
                   const int tmp_size,
                   const int tw,
                   const float4 homo_1st,
                   const float4 homo_2nd,
                   const int iw,
                   const int ih,
                   float *img_y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tmp_size)
        return;

    int y = idx / tw;
    int x = idx % tw;

    float inv_z = 1 / (homo_1st.z*x + homo_2nd.y*y + 1);
    int u = int((homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_z + 0.5f);
    int v = int((homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_z + 0.5f);
    u = u < 0 ? 0 : u > iw - 1 ? iw - 1 : u;
    v = v < 0 ? 0 : v > ih - 1 ? ih - 1 : v;

    img_y[idx] = img(v, u).x;
}

__global__
void normalizeImgYKernel(cv::cuda::PtrStepSz<float3> img,
                         const int img_size, 
                         const int iw,
                         const float alpha,
                         const float beta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= img_size)
        return;

    int v = idx / iw;
    int u = idx % iw;
    img(v, u).x = alpha * (img(v, u).x) + beta;
}

}  // namespace pr
