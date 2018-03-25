#include <cuda.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <cuda_texture_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mex.h>
#include "coarseToFinePoseEsti.h"
#include "apeCommon.h"

namespace ape {

static const int BLOCK_W = 8;
static const int BLOCK_H = 8;
static const int BLOCK_SIZE_2D = BLOCK_W*BLOCK_H;
static const int BLOCK_SIZE = 256;
static const int ORI_SAMPLE_NUM = 448;

// constant memory
__constant__ float2 const_tmp_coors[ORI_SAMPLE_NUM];
__constant__ float3 const_tmp_vals[ORI_SAMPLE_NUM];


// texture memory
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_img;

void coarseToFinePoseEstimation(const cv::cuda::GpuMat &tmp,
                                const cv::cuda::GpuMat &img,
                                int prm_lvls,
                                bool photo_inva,
                                bool verbose,
                                ApeParams *ape_params,
                                double *ex_mat) {
    // bind texture memory
    tex_img.addressMode[0] = cudaAddressModeBorder;
    tex_img.addressMode[1] = cudaAddressModeBorder;
    tex_img.filterMode = cudaFilterModeLinear;
    tex_img.normalized = false;
    cudaChannelFormatDesc cuda_channel_format_desc = cudaCreateChannelDesc<float4>();

    // poses
    size_t num_poses;
    thrust::device_vector<Pose> poses;
    thrust::device_vector<float> dists;
    thrust::host_vector<float> min_dists;

    // parameters
    const float2 tmp_real = make_float2(ape_params->tmp_real_w, ape_params->tmp_real_h);
    const float factor = 1 / 1.511f;
    const float2 constraint = (photo_inva) ? make_float2(0.075f, 0.15f) : make_float2(0.05f, 0.1f);
    int level = 0;
    int level_p = 0;
    
    float begin_scale = 1.f/powf(4.f, prm_lvls-1.f);
    for (float scale = begin_scale; scale <= 1; scale *= 4) {
        if (verbose) {
            mexPrintf("pyramid: %f\n", scale);
            mexEvalString("drawnow;");
        }
        int sample_num = ORI_SAMPLE_NUM;
        // rescale image
        cv::cuda::GpuMat small_img, small_tmp;
        ApeParams small_ape_params;
        rescale(tmp, img, *ape_params, scale, &small_tmp, &small_img, &small_ape_params);
        cudaBindTexture2D(0, &tex_img, small_img.data, &cuda_channel_format_desc,
                          small_ape_params.iw, small_ape_params.ih, small_img.step);

        // allocate sample memory
        thrust::device_vector<float2> tmp_coors(sample_num, make_float2(0, 0));
        thrust::device_vector<float3> tmp_vals(sample_num, make_float3(0, 0, 0));
        int2 tmp_dim = make_int2(small_ape_params.tw, small_ape_params.th);

        while (true) {
            // initialize the net
            if (level == 0)
                createSet(small_ape_params, &poses, &num_poses);

            ++level;
            randSample(small_tmp, tmp_dim, tmp_real, sample_num, &tmp_coors, &tmp_vals);
            dists.resize(num_poses);
            size_t ori_num_poses = num_poses;
            calDist(poses,
                    make_float4(small_ape_params.fx, small_ape_params.fy, small_ape_params.cx, small_ape_params.cy),
                    tmp_real,
                    make_int2(small_ape_params.iw, small_ape_params.ih),
                    photo_inva,
                    num_poses,
                    sample_num,
                    &dists);
            cudaDeviceSynchronize();
            auto dists_iter = thrust::min_element(dists.begin(), dists.end());
            float min_dist = *dists_iter;
            if (verbose) {
                mexPrintf("  -- level %d -- epsilon %.3f, Number of Poses %d, Minimum Dist. %f\n",
                          level, small_ape_params.epsilon, num_poses, min_dist);
                mexEvalString("drawnow;");
            }
            min_dists.push_back(min_dist);

            // early termination
            if ((min_dist < 0.005) || ((scale == 1) && (min_dist < 0.015)) || 
                ((level_p > 0) && (level_p != level) && (scale == 1) && (min_dist > calLastThreeTermsMean(min_dists, level-level_p)*0.97))) {
                auto idx = dists_iter - dists.begin();
                getExMat(poses[idx], ex_mat);
                cudaUnbindTexture(&tex_img);
                return;
            }

            // get poses by distance
            bool too_high_percentage = getPosesByDistance(dists,
                                                          min_dist,
                                                          small_ape_params.epsilon,
                                                          &poses,
                                                          &num_poses);

            // expand the pose set for next round
            // if the initial pose set is not decent enough, recreate another new epsilon-covering set with smaller epsilon
            if ((level == 1)
                && ((too_high_percentage && (min_dist > constraint.x) && (ori_num_poses < 7500000))
                    || ((min_dist > constraint.y) && (ori_num_poses < 5000000)))) {
                small_ape_params.ShrinkNet(0.9f);
                level = 0;
                min_dists.clear();
            }
            else {
                expandPoses(factor, &poses, &small_ape_params, &num_poses);
                thrust::device_vector<float> tzs(num_poses, 0.0);
                const size_t BLOCK_NUM = ((num_poses) - 1) / BLOCK_SIZE + 1;
                fetchTzKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(thrust::raw_pointer_cast(poses.data()),
                                                         num_poses,
                                                         thrust::raw_pointer_cast(tzs.data()));
                float pixelMaxMovement =  small_ape_params.epsilon
                                          * std::max(small_ape_params.fx, small_ape_params.fy)
                                          * std::max(small_ape_params.tmp_real_w, small_ape_params.tmp_real_h)
                                          * 2
                                          / (thrust::reduce(tzs.begin(), tzs.end()) / num_poses);
                if (pixelMaxMovement < 1) {
                    level_p = level + 1;
                    ape_params->UpdateNet(small_ape_params);
                    break;
                }
            }
        }
        cudaUnbindTexture(&tex_img);
    }
    calDist(poses,
            make_float4(ape_params->fx, ape_params->fy, ape_params->cx, ape_params->cy),
            tmp_real,
            make_int2(ape_params->iw, ape_params->ih),
            photo_inva,
            num_poses,
            ORI_SAMPLE_NUM,
            &dists);
    cudaDeviceSynchronize();
    auto dists_iter = thrust::min_element(dists.begin(), dists.end());
    auto idx = dists_iter - dists.begin();
    getExMat(poses[idx], ex_mat);
}

void rescale(const cv::cuda::GpuMat &tmp,
             const cv::cuda::GpuMat &img,
             const ApeParams &ape_params,
             const float scale,
             cv::cuda::GpuMat *small_tmp,
             cv::cuda::GpuMat *small_img,
             ApeParams *small_ape_params) {
    cv::cuda::resize(img, *small_img, cv::Size(), scale, scale, CV_INTER_AREA);
    cv::cuda::resize(tmp, *small_tmp, cv::Size(), scale, scale, CV_INTER_AREA);
    *small_ape_params = ape_params;
    small_ape_params->iw = small_img->cols;
    small_ape_params->ih = small_img->rows;
    small_ape_params->tw = small_tmp->cols;
    small_ape_params->th = small_tmp->rows;

    // modify intrinsic papameters
    const float offset = -0.5f;
    small_ape_params->fx = ape_params.fx * scale;
    small_ape_params->fy = ape_params.fy * scale;
    small_ape_params->cx = (ape_params.cx - offset) * scale + offset;
    small_ape_params->cy = (ape_params.cy - offset) * scale + offset;

    float tz_square = ape_params.max_tz * ape_params.min_tz;
    float area =  (2 * small_ape_params->fx * ape_params.tmp_real_w)
                * (2 * small_ape_params->fy * ape_params.tmp_real_h) / tz_square;
    float length = sqrt(area);
    float total_variation = getTotalVariation(*small_tmp, small_ape_params->tw, small_ape_params->th, area);
    cv::Ptr<cv::cuda::Filter> gaussian_filter_tmp = cv::cuda::createGaussianFilter(CV_32FC3, CV_32FC3, cv::Size(5, 5), 1);
    cv::Ptr<cv::cuda::Filter> gaussian_filter_img = cv::cuda::createGaussianFilter(CV_32FC4, CV_32FC4, cv::Size(5, 5), 1);
    while (total_variation > 8.42 * length) { // 8.42 is obtained emperically
        gaussian_filter_tmp->apply(*small_tmp, *small_tmp);
        gaussian_filter_img->apply(*small_img, *small_img);
        total_variation = getTotalVariation(*small_tmp, small_ape_params->tw, small_ape_params->th, area);
    }
}

float getTotalVariation(const cv::cuda::GpuMat &tmp,
                        int tw,
                        int th,
                        float area) {
    // allocate
    const int pxl_num = tw * th;
    thrust::device_vector<float> variation(pxl_num, 0.0);

    // kernel parameter for TV
    dim3 b_dim(BLOCK_W, BLOCK_H);
    dim3 g_dim((tw - 1) / BLOCK_W + 1, (th - 1) / BLOCK_H + 1);
    variationKernel<<<g_dim, b_dim>>>(tmp, make_int2(tw, th), thrust::raw_pointer_cast(variation.data()));
    float total_variation = thrust::reduce(variation.begin(), variation.end()) / pxl_num * area;
    return total_variation;
}

__global__
void variationKernel(const cv::cuda::PtrStepSz<float3> tmp,
                     int2 dim,
                     float* variation) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * blockDim.x + tidx;
    const int x = blockIdx.x * blockDim.x + tidx;
    const int y = blockIdx.y * blockDim.y + tidy;

    // store pixels in specific window in shared memory
    // the window is expanded by on pixel from block
    const int ww = BLOCK_W + 2;
    const int wh = BLOCK_H + 2;
    const int ws = ww * wh;
    __shared__ float window[ws];

    // move data to the shared memory
    int x_begin = blockIdx.x * blockDim.x - 1;
    int y_begin = blockIdx.y * blockDim.y - 1;
    for (int i = tid; i < ws; i += BLOCK_SIZE_2D) {
        int wx = (i % ww) + x_begin;
        int wy = (i / ww) + y_begin;
        if (wx < 0 || wx >= dim.x || wy < 0 || wy >= dim.y)
            window[i] = 2;
        else
            window[i] = tmp(wy, wx).x;
    }
    __syncthreads();

    // out of range
    if (x >= dim.x || y >= dim.y)
        return;

    // find max difference between center pixel and surroundings
    float max_diff = 0;
    float value = window[(tidy + 1)*ww + (tidx + 1)];
    for (int idy = 0, wi = tidy*ww + tidx; idy < 3; ++idy, wi += (BLOCK_W - 1)) {
        for (int idx = 0; idx < 3; ++idx, ++wi) {
            float surr = window[wi];
            if (surr != 2) {
                float diff = std::abs(value - surr);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }
    }
    variation[y*dim.x + x] = max_diff;
}

void randSample(const cv::cuda::GpuMat &tmp,
                int2 tmp_dim,
                float2 tmp_real,
                int sample_num,
                thrust::device_vector<float2> *tmp_coors,
                thrust::device_vector<float3> *tmp_vals) {
    // rand pixel
    srand(time(NULL));
    thrust::device_vector<int2> rand_coor(sample_num, make_int2(0, 0));
    thrust::counting_iterator<int> i0(rand()/2);
    thrust::transform(i0, i0 + sample_num, rand_coor.begin(), CoorRngFunctor(tmp_dim.x, tmp_dim.y));

    // get pixel value and position
    const int BLOCK_NUM = (sample_num - 1) / BLOCK_SIZE + 1;
    randSampleKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(thrust::raw_pointer_cast(rand_coor.data()),
                                                tmp,
                                                tmp_dim,
                                                tmp_real,
                                                sample_num,
                                                thrust::raw_pointer_cast(tmp_coors->data()),
                                                thrust::raw_pointer_cast(tmp_vals->data()));

    // bind to const mem
    cudaMemcpyToSymbol(const_tmp_coors, thrust::raw_pointer_cast(tmp_coors->data()), sizeof(float2)* sample_num, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(const_tmp_vals, thrust::raw_pointer_cast(tmp_vals->data()), sizeof(float3)* sample_num, 0, cudaMemcpyDeviceToDevice);
}

__global__
void randSampleKernel(const int2 *rand_coor,
                      const cv::cuda::PtrStepSz<float3> tmp,
                      int2 tmp_dim,
                      float2 tmp_real,
                      int sample_num,
                      float2 *tmp_coors,
                      float3 *tmp_vals) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_num)
        return;

    int x = rand_coor[idx].x;
    int y = rand_coor[idx].y;

    tmp_vals[idx] = tmp(y, x);

    float2 coor;
    coor.x = (2 * float(x) + 1 - tmp_dim.x) / tmp_dim.x * tmp_real.x;
    coor.y = -(2 * float(y) + 1 - tmp_dim.y) / tmp_dim.y * tmp_real.y;
    tmp_coors[idx] = coor;
}

void createSet(const ApeParams &ape_params,
               thrust::device_vector<Pose> *poses,
               size_t *num_poses) {
    // count
    int count_total = 0;
    thrust::host_vector<int4> count; // rz0 rz1 tx ty

    // paramters
    const float length = sqrt(ape_params.tmp_real_w * ape_params.tmp_real_w + ape_params.tmp_real_h * ape_params.tmp_real_h);
    const int NUM_RZ0 = int((ape_params.max_rz - ape_params.min_rz) / ape_params.step.rz0) + 1;
    const int NUM_RZ1 = int((ape_params.max_rz - ape_params.min_rz) / ape_params.step.rz1) + 1;

    // counting
    for (float tz = ape_params.min_tz; tz <= ape_params.max_tz; ) {
        float bound_tx = fabs(ape_params.cx*tz / ape_params.fx - ape_params.tmp_real_h);
        float bound_ty = fabs(ape_params.cy*tz / ape_params.fy - ape_params.tmp_real_h);
        for (float rx = ape_params.min_rx; rx <= ape_params.max_rx; ) {
            int num_rz0 = (rx != 0) ? NUM_RZ0 : 1;
            int num_tx = int(2 * bound_tx / (ape_params.step.tx*(tz - length*sin(rx)))) + 1;
            int num_ty = int(2 * bound_ty / (ape_params.step.ty*(tz - length*sin(rx)))) + 1;
            count_total += (num_rz0 * NUM_RZ1 * num_tx * num_ty);
            count.push_back(make_int4(num_rz0, NUM_RZ1, num_tx, num_ty));
            float asin_value = 2 - 1 / (1 / (2 - sin(rx)) + ape_params.step.rx);
            if (asin_value <= 1 && asin_value >= -1)
                rx = asinf(asin_value);
            else
                rx = ape_params.max_rx + 1;
        }
        tz += tz*tz*ape_params.step.tz / (1 - ape_params.step.tz*tz);
    }

    // allocate
    thrust::device_vector<bool> valids(count_total, false);
    poses->resize(count_total);
    
    // assignment
    Pose* poses_ptr = thrust::raw_pointer_cast(poses->data());
    bool* valids_ptr = thrust::raw_pointer_cast(valids.data());
    auto count_iter = count.begin();
    int begin_index = 0;
    int area_thres = std::round(0.01 * ape_params.iw * ape_params.ih);
    for (float tz = ape_params.min_tz; tz <= ape_params.max_tz; ) {
        float bound_tx = fabs(ape_params.cx*tz / ape_params.fx - ape_params.tmp_real_h);
        float bound_ty = fabs(ape_params.cy*tz / ape_params.fy - ape_params.tmp_real_h);
        for (float rx = ape_params.min_rx; rx <= ape_params.max_rx; ) {
            int num = (*count_iter).x * (*count_iter).y * (*count_iter).z * (*count_iter).w;
            const int BLOCK_NUM = (num - 1) / BLOCK_SIZE + 1;
            float2 bound = make_float2(bound_tx, bound_ty);
            createSetKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(begin_index,
                                                       *count_iter,
                                                       num,
                                                       tz,
                                                       rx,
                                                       ape_params,
                                                       bound,
                                                       length,
                                                       area_thres,
                                                       poses_ptr,
                                                       valids_ptr);
            begin_index += num;
            ++count_iter;

            float asin_value = 2 - 1 / (1 / (2 - sin(rx)) + ape_params.step.rx);
            if (asin_value <= 1 && asin_value >= -1)
                rx = asinf(asin_value);
            else
                rx = ape_params.max_rx + 1;
        }
        tz += tz*tz*ape_params.step.tz / (1 - ape_params.step.tz*tz);
    }

    if (begin_index != count_total)
        std::cerr << "error occur in 'createSet'!" << std::endl;

    // remove non-valid poses
    auto zip_it_valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(poses->begin(), valids.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(poses->end(), valids.end())),
        ValidFunctor()
    );
    poses->erase(thrust::get<0>(zip_it_valid_end.get_iterator_tuple()), poses->end());
    *num_poses = poses->size();
}

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
                     bool *valids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num)
        return;

    const int num_rz0 = count.x;
    const int num_rz1 = count.y;
    const int num_tx = count.z;
    const int num_ty = count.w;   

    const int id_rz0 = idx % num_rz0;
    const int id_rz1 = (idx / num_rz0) % num_rz1;
    const int id_ty = (idx / (num_rz0 * num_rz1)) % num_ty;
    const int id_tx = (idx / (num_rz0 * num_rz1 * num_ty)) % num_tx;

    Pose pose;
    pose.rz0 = ape_params.min_rz + id_rz0*ape_params.step.rz0;
    pose.rx = rx;
    pose.rz1 = ape_params.min_rz + id_rz1*ape_params.step.rz1;
    pose.tx = -bound.x + id_tx*ape_params.step.tx*(tz - length*sinf(rx));
    pose.ty = -bound.y + id_ty*ape_params.step.ty*(tz - length*sinf(rx));
    pose.tz = tz;

    int index = idx + begin_index;
    poses[index] = pose;

    // calculate homography parameters
    pose.rx += 3.1415926f;

    // pre-compute sin and cos values
    float cos_rz0 = cosf(pose.rz0);
    float cos_rx = cosf(pose.rx);
    float cos_rz1 = cosf(pose.rz1);
    float sin_rz0 = sinf(pose.rz0);
    float sin_rx = sinf(pose.rx);
    float sin_rz1 = sinf(pose.rz1);

    //  z coordinate is y cross x, so add minus
    float r11 = cos_rz0 * cos_rz1 - sin_rz0 * cos_rx * sin_rz1;
    float r12 = -cos_rz0 * sin_rz1 - sin_rz0 * cos_rx * cos_rz1;
    float r21 = sin_rz0 * cos_rz1 + cos_rz0 * cos_rx * sin_rz1;
    float r22 = -sin_rz0 * sin_rz1 + cos_rz0 * cos_rx * cos_rz1;
    float r31 = sin_rx * sin_rz1;
    float r32 = sin_rx * cos_rz1;

    // final transfomration
    float t0 = ape_params.fx*r11 + ape_params.cx*r31;
    float t1 = ape_params.fx*r12 + ape_params.cx*r32;
    float t3 = ape_params.fx*pose.tx + ape_params.cx*pose.tz;
    float t4 = ape_params.fy*r21 + ape_params.cy*r31;
    float t5 = ape_params.fy*r22 + ape_params.cy*r32;
    float t7 = ape_params.fy*pose.ty + ape_params.cy*pose.tz;
    float t8 = r31;
    float t9 = r32;
    float t11 = pose.tz;

    // reject transformations make template out of boundary
    float inv_c1z = 1 / (t8*(-ape_params.tmp_real_w) + t9*(-ape_params.tmp_real_h) + t11);
    float c1x = (t0*(-ape_params.tmp_real_w) + t1*(-ape_params.tmp_real_h) + t3) * inv_c1z;
    float c1y = (t4*(-ape_params.tmp_real_w) + t5*(-ape_params.tmp_real_h) + t7) * inv_c1z;
    float inv_c2z = 1 / (t8*(+ape_params.tmp_real_w) + t9*(-ape_params.tmp_real_h) + t11);
    float c2x = (t0*(+ape_params.tmp_real_w) + t1*(-ape_params.tmp_real_h) + t3) * inv_c2z;
    float c2y = (t4*(+ape_params.tmp_real_w) + t5*(-ape_params.tmp_real_h) + t7) * inv_c2z;
    float inv_c3z = 1 / (t8*(+ape_params.tmp_real_w) + t9*(+ape_params.tmp_real_h) + t11);
    float c3x = (t0*(+ape_params.tmp_real_w) + t1*(+ape_params.tmp_real_h) + t3) * inv_c3z;
    float c3y = (t4*(+ape_params.tmp_real_w) + t5*(+ape_params.tmp_real_h) + t7) * inv_c3z;
    float inv_c4z = 1 / (t8*(-ape_params.tmp_real_w) + t9*(+ape_params.tmp_real_h) + t11);
    float c4x = (t0*(-ape_params.tmp_real_w) + t1*(+ape_params.tmp_real_h) + t3) * inv_c4z;
    float c4y = (t4*(-ape_params.tmp_real_w) + t5*(+ape_params.tmp_real_h) + t7) * inv_c4z;
    float minx = fminf(c1x, fminf(c2x, fminf(c3x, c4x)));
    float maxx = fmaxf(c1x, fmaxf(c2x, fmaxf(c3x, c4x)));
    float miny = fminf(c1y, fminf(c2y, fminf(c3y, c4y)));
    float maxy = fmaxf(c1y, fmaxf(c2y, fmaxf(c3y, c4y)));

    // reject transformations make marker too small in screen
    float two_area =  (c1x - c2x) * (c1y + c2y)
                    + (c2x - c3x) * (c2y + c3y)
                    + (c3x - c4x) * (c3y + c4y)
                    + (c4x - c1x) * (c4y + c1y);
    float area = abs(two_area / 2);

    const int margin = 1;
    if (area > area_thres
        && (minx >= margin)
        && (maxx <= ape_params.iw -1 - margin)
        && (miny >= margin)
        && (maxy <= ape_params.ih -1 - margin))
        valids[index] = true;
    else
        valids[index] = false;
}

void calDist(const thrust::device_vector<Pose> &poses,
             float4 in_params,
             float2 tmp_real,
             int2 img_dim,
             bool photo_inva,
             size_t num_poses,
             size_t sample_num,
             thrust::device_vector<float> *dists) {
    const size_t BLOCK_NUM = (num_poses - 1) / BLOCK_SIZE + 1;
    float inv_sample_num = 1.f / sample_num;
    if (photo_inva) {
        thrust::device_vector<float> tmp_y(sample_num);
        getTmpY<<<int((sample_num - 1) / BLOCK_SIZE + 1), BLOCK_SIZE >>>(thrust::raw_pointer_cast(tmp_y.data()), sample_num);
        float sum_of_tmp_y = thrust::reduce(tmp_y.begin(), tmp_y.end());
        float mean_tmp = sum_of_tmp_y * inv_sample_num;
        float sum_of_sq_tmp_y = thrust::inner_product(thrust::device, tmp_y.begin(), tmp_y.end(), tmp_y.begin(), 0.f);
        float sig_tmp = sqrt(fmaxf((sum_of_sq_tmp_y - (sum_of_tmp_y*sum_of_tmp_y) * inv_sample_num), 0.f) * inv_sample_num) + 1e-7f;
        calDistInvarKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(thrust::raw_pointer_cast(poses.data()),
                                                      in_params,
                                                      tmp_real,
                                                      img_dim,
                                                      num_poses,
                                                      sample_num,
                                                      inv_sample_num,
                                                      mean_tmp,
                                                      sig_tmp,
                                                      thrust::raw_pointer_cast(dists->data()));
    }
    else {
        calDistColorKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(thrust::raw_pointer_cast(poses.data()),
                                                      in_params,
                                                      tmp_real,
                                                      img_dim,
                                                      num_poses,
                                                      sample_num,
                                                      inv_sample_num,
                                                      thrust::raw_pointer_cast(dists->data()));
    }
}

__global__
void getTmpY(float *tmp_y,
             int sample_num) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_num)
        return;

    tmp_y[idx] = const_tmp_vals[idx].x;
}

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
                        float *dists) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_poses)
        return;

    // get pose parameter
    float rz0 = poses[idx].rz0;
    float rx = poses[idx].rx + 3.1415926f;
    float rz1 = poses[idx].rz1;
    float tx = poses[idx].tx;
    float ty = poses[idx].ty;
    float tz = poses[idx].tz;

    float cos_rz0 = cosf(rz0);     
    float cos_rx = cosf(rx); 
    float cos_rz1 = cosf(rz1);
    float sin_rz0 = sinf(rz0);
    float sin_rx = sinf(rx);
    float sin_rz1 = sinf(rz1);

    // z coordinate is y cross x, so add minus
    float r11 = cos_rz0 * cos_rz1 - sin_rz0 * cos_rx * sin_rz1;
    float r12 = -cos_rz0 * sin_rz1 - sin_rz0 * cos_rx * cos_rz1;
    float r21 = sin_rz0 * cos_rz1 + cos_rz0 * cos_rx * sin_rz1;
    float r22 = -sin_rz0 * sin_rz1 + cos_rz0 * cos_rx * cos_rz1;
    float r31 = sin_rx * sin_rz1;
    float r32 = sin_rx * cos_rz1;

    // final transfomration
    float t0 = in_params.x*r11 + in_params.z*r31;
    float t1 = in_params.x*r12 + in_params.z*r32;
    float t3 = in_params.x*tx + in_params.z*tz;
    float t4 = in_params.y*r21 + in_params.w*r31;
    float t5 = in_params.y*r22 + in_params.w*r32;
    float t7 = in_params.y*ty + in_params.w*tz;
    float t8 = r31;
    float t9 = r32;
    float t11 = tz;

    // calculate distance
    float score = 0.f;

    // parameters for normalization
    float sum_of_img_y = 0;
    float sum_of_sq_img_y = 0;
    for (int i = 0; i < sample_num; ++i) {
        // calculate coordinate on camera image
        float inv_z = 1 / (t8*const_tmp_coors[i].x + t9*const_tmp_coors[i].y + t11);
        float u = (t0*const_tmp_coors[i].x + t1*const_tmp_coors[i].y + t3) * inv_z;
        float v = (t4*const_tmp_coors[i].x + t5*const_tmp_coors[i].y + t7) * inv_z;

        // get value from constant memory
        float3 tmp_val = const_tmp_vals[i];

        // get value from texture
        // have to add 0.5f for coordinates (see E.2 Linear Filtering in CUDA Programming Guide)
        float4 img_val = tex2D(tex_img, u + 0.5f, v + 0.5f);

        // accumulation for normalization
        sum_of_img_y += img_val.x;
        sum_of_sq_img_y += img_val.x*img_val.x;

        float inv_num = 1.f / (i + 1);
        float sig_img = sqrt(fmaxf((sum_of_sq_img_y - (sum_of_img_y*sum_of_img_y) * inv_num), 0.f) * inv_num) + 1e-7f;
        float mean_img = sum_of_img_y * inv_num;
        float sig_tmp_over_sig_img = sig_tmp / sig_img;
        float faster = -mean_tmp + sig_tmp_over_sig_img * mean_img;

        // calculate distant
        score += 0.50f * abs(tmp_val.x - sig_tmp_over_sig_img*img_val.x + faster)
               + 0.25f * abs(img_val.y - tmp_val.y)
               + 0.25f  *abs(img_val.z - tmp_val.z);
    }

    dists[idx] = score * inv_sample_num;
}

__global__
void calDistColorKernel(const Pose *poses,
                        float4 in_params,
                        float2 tmp_real,
                        int2 img_dim,
                        size_t num_poses,
                        size_t sample_num,
                        float inv_sample_num,
                        float *dists) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_poses)
        return;

    // get pose parameter
    float rz0 = poses[idx].rz0;
    float rx = poses[idx].rx + 3.1415926f;
    float rz1 = poses[idx].rz1;
    float tx = poses[idx].tx;
    float ty = poses[idx].ty;
    float tz = poses[idx].tz;

    float cos_rz0 = cosf(rz0); 
    float cos_rx = cosf(rx);
    float cos_rz1 = cosf(rz1); 
    float sin_rz0 = sinf(rz0);
    float sin_rx = sinf(rx);
    float sin_rz1 = sinf(rz1);

    //  z coordinate is y cross x, so add minus
    float r11 = cos_rz0 * cos_rz1 - sin_rz0 * cos_rx * sin_rz1;
    float r12 = -cos_rz0 * sin_rz1 - sin_rz0 * cos_rx * cos_rz1;
    float r21 = sin_rz0 * cos_rz1 + cos_rz0 * cos_rx * sin_rz1;
    float r22 = -sin_rz0 * sin_rz1 + cos_rz0 * cos_rx * cos_rz1;
    float r31 = sin_rx * sin_rz1;
    float r32 = sin_rx * cos_rz1;

    // final transfomration
    float t0 = in_params.x*r11 + in_params.z*r31;
    float t1 = in_params.x*r12 + in_params.z*r32;
    float t3 = in_params.x*tx + in_params.z*tz;
    float t4 = in_params.y*r21 + in_params.w*r31;
    float t5 = in_params.y*r22 + in_params.w*r32;
    float t7 = in_params.y*ty + in_params.w*tz;
    float t8 = r31;
    float t9 = r32;
    float t11 = tz;

    // calculate distance
    float score = 0.0;
    for (int i = 0; i < sample_num; ++i) {
        // calculate coordinate on camera image
        float inv_z = 1 / (t8*const_tmp_coors[i].x + t9*const_tmp_coors[i].y + t11);
        float u = (t0*const_tmp_coors[i].x + t1*const_tmp_coors[i].y + t3) * inv_z;
        float v = (t4*const_tmp_coors[i].x + t5*const_tmp_coors[i].y + t7) * inv_z;

        // get value from constant memory
        float3 tmp_val = const_tmp_vals[i];

        // get value from texture
        // have to add 0.5f for coordinates (see E.2 Linear Filtering in CUDA Programming Guide)
        float4 img_val = tex2D(tex_img, u + 0.5f, v + 0.5f);

        // calculate distant
        score += 0.50 * abs(img_val.x - tmp_val.x)
               + 0.25 * abs(img_val.y - tmp_val.y)
               + 0.25 * abs(img_val.z - tmp_val.z);
    }
    dists[idx] = score * inv_sample_num;
}

float calLastThreeTermsMean(thrust::host_vector<float> &min_dists,
                            int iter_times) {
    float sum = 0;
    int count = 0;
    iter_times = (iter_times < 3) ? iter_times : 3;
    for (auto it = min_dists.rbegin(); it != min_dists.rend() && count < iter_times; ++it, ++count) {
        sum += *it;
    }
    return sum / count;
}


void getExMat(const Pose &pose,
              double *ex_mat) {
    float rz0 = pose.rz0;
    float rx = pose.rx + 3.1415926f;
    float rz1 = pose.rz1;

    float sin_rz0 = sin(rz0);
    float cos_rz0 = cos(rz0);
    float sin_rx = sin(rx);
    float cos_rx = cos(rx);
    float sin_rz1 = sin(rz1);
    float cos_rz1 = cos(rz1);

    ex_mat[0] = double(cos_rz0*cos_rz1 - sin_rz0*cos_rx*sin_rz1);
    ex_mat[4] = double(-cos_rz0*sin_rz1 - sin_rz0*cos_rx*cos_rz1);
    ex_mat[8] = double(sin_rz0*sin_rx);
    ex_mat[12] = double(pose.tx);
    ex_mat[1] = double(sin_rz0*cos_rz1 + cos_rz0*cos_rx*sin_rz1);
    ex_mat[5] = double(-sin_rz0*sin_rz1 + cos_rz0*cos_rx*cos_rz1);
    ex_mat[9] = double(-cos_rz0*sin_rx);
    ex_mat[13] = double(pose.ty);
    ex_mat[2] = double(sin_rx*sin_rz1);
    ex_mat[6] = double(sin_rx*cos_rz1);
    ex_mat[10] = double(cos_rx);
    ex_mat[14] = double(pose.tz);
    ex_mat[3] = 0.0;
    ex_mat[7] = 0.0;
    ex_mat[11] = 0.0;
    ex_mat[15] = 1.0;
}

bool getPosesByDistance(const thrust::device_vector<float> &dists,
                        float min_dist,
                        float epsilon,
                        thrust::device_vector<Pose> *poses,
                        size_t *num_poses) {
    // get initial threhold
    const float threshold = 0.19 * epsilon + 0.01372;
    min_dist += threshold;

    // count reductions
    bool too_high_percentage = false;
    bool first = true;
    size_t count = INT_MAX;
    thrust::device_vector<bool> survivals(*num_poses, false);
    const size_t BLOCK_NUM = (*num_poses - 1) / BLOCK_SIZE + 1;
    while (true) {
        getPosesByDistanceKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(thrust::raw_pointer_cast(dists.data()),
                                                            min_dist,
                                                            dists.size(),
                                                            thrust::raw_pointer_cast(survivals.data()));
        count = thrust::count(survivals.begin(), survivals.end(), true);
        if (first) {
            float percentage = float(count) / *num_poses;
            too_high_percentage = (percentage > 0.1f);
            first = false;
        }
        // reduce the size of pose set to prevent from out of memory
        if (count < 27000) {
            if (count == 0) {
                auto dists_iter = thrust::min_element(dists.begin(), dists.end());
                unsigned int position = dists_iter - dists.begin();
                thrust::device_vector<Pose> temp_pose(1);
                temp_pose[0] = (*poses)[position];
                *poses = temp_pose;
                *num_poses = 1;
            }
            else {
                // prune poses
                auto zip_it_valid_end = thrust::remove_if(
                    thrust::make_zip_iterator(thrust::make_tuple(poses->begin(), survivals.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(poses->end(), survivals.end())),
                    ValidFunctor()
                );
                poses->erase(thrust::get<0>(zip_it_valid_end.get_iterator_tuple()), poses->end());
                *num_poses = count;
            }
            break;
        }
        min_dist *= 0.99f;
    }
    return too_high_percentage;
}

__global__
void getPosesByDistanceKernel(const float *dists,
                              float threshold,
                              size_t num_poses,
                              bool *survivals) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_poses)
        return;

    survivals[idx] = (dists[idx] < threshold) ? true : false;
}

void expandPoses(float factor,
                 thrust::device_vector<Pose> *poses,
                 ApeParams *ape_params,
                 size_t *num_poses) {
    // number of expand points
    const int multiple = 80;
    size_t new_num_poses = (*num_poses) * (multiple + 1);

    // decrease step
    ape_params->ShrinkNet(factor);

    // expand origin set
    const size_t BLOCK_NUM = ((*num_poses) - 1) / BLOCK_SIZE + 1;
    int area_thres = 0.01 * ape_params->iw * ape_params->ih;
    thrust::device_vector<bool> valids(new_num_poses, true);
    poses->resize(new_num_poses);
    expandPosesKernel<<<BLOCK_NUM, BLOCK_SIZE>>>(*num_poses,
                                                 new_num_poses,
                                                 *ape_params,
                                                 area_thres,
                                                 thrust::raw_pointer_cast(poses->data()),
                                                 thrust::raw_pointer_cast(valids.data()));

    // remove invalid terms
    auto zip_it_valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(poses->begin(), valids.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(poses->end(), valids.end())),
        ValidFunctor()
    );
    poses->erase(thrust::get<0>(zip_it_valid_end.get_iterator_tuple()), poses->end());
    *num_poses = poses->size();
}

__global__
void expandPosesKernel(size_t num_poses,
                       size_t new_num_poses,
                       ApeParams ape_params,
                       int area_thres,
                       Pose *poses,
                       bool *valids) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_poses)
        return;

    curandState_t state;
    curand_init(idx, 0, 0, &state);

    float ori_rz0 = poses[idx].rz0;
    float ori_rx = poses[idx].rx;
    float ori_rz1 = poses[idx].rz1;
    float ori_tx = poses[idx].tx;
    float ori_ty = poses[idx].ty;
    float ori_tz = poses[idx].tz;
    for (unsigned int i = idx + num_poses; i < new_num_poses; i += num_poses) {
        // rz0
        Pose pose;
        pose.rz0 = ori_rz0 + (curand(&state) % 3 - 1.f)*ape_params.step.rz0;

        // rx
        float is_plus = (curand(&state) % 3 - 1.f);
        float sin_ori_rx = 2 - 1 / (1 / (2 - sinf(ori_rx)) + is_plus*ape_params.step.rx);
        pose.rx = ori_rx + is_plus * is_plus * (asinf(sin_ori_rx) - ori_rx);

        // rz1
        pose.rz1 = ori_rz1 + (curand(&state) % 3 - 1.f)*ape_params.step.rz1;

        // tx ty
        float weight = ori_tz + sqrt(ape_params.tmp_real_w*ape_params.tmp_real_w + ape_params.tmp_real_h*ape_params.tmp_real_h) * sinf(ori_rx);
        pose.tx = ori_tx + (curand(&state) % 3 - 1.f) * weight * ape_params.step.tx;
        pose.ty = ori_ty + (curand(&state) % 3 - 1.f) * weight * ape_params.step.ty;

        // tz
        is_plus = (curand(&state) % 3 - 1.f);
        float denom_tz = 1 - is_plus * ape_params.step.tz * ori_tz;
        pose.tz = ori_tz + is_plus * ape_params.step.tz * (ori_tz * ori_tz) / denom_tz;

        poses[i] = pose;

        // condition
        bool valid = (denom_tz != 0)
            & (abs(sin_ori_rx) <= 1)
            & (pose.tz >= ape_params.min_tz)
            & (pose.tz <= ape_params.max_tz)
            & (pose.rx >= ape_params.min_rx)
            & (pose.rx <= ape_params.max_rx);

        if (valid == false) {
            valids[i] = false;
            return;
        }

        // calculate homography parameters
        pose.rx += 3.1415926;

        // pre-compute sin and cos values
        float cos_rz0 = cosf(pose.rz0);
        float cos_rx = cosf(pose.rx);
        float cos_rz1 = cosf(pose.rz1);
        float sin_rz0 = sinf(pose.rz0);
        float sin_rx = sinf(pose.rx);
        float sin_rz1 = sinf(pose.rz1);

        //  z coordinate is y cross x, so add minus
        float r11 = cos_rz0 * cos_rz1 - sin_rz0 * cos_rx * sin_rz1;
        float r12 = -cos_rz0 * sin_rz1 - sin_rz0 * cos_rx * cos_rz1;
        float r21 = sin_rz0 * cos_rz1 + cos_rz0 * cos_rx * sin_rz1;
        float r22 = -sin_rz0 * sin_rz1 + cos_rz0 * cos_rx * cos_rz1;
        float r31 = sin_rx * sin_rz1;
        float r32 = sin_rx * cos_rz1;

        // final transfomration
        float t0 = ape_params.fx*r11 + ape_params.cx*r31;
        float t1 = ape_params.fx*r12 + ape_params.cx*r32;
        float t3 = ape_params.fx*pose.tx + ape_params.cx*pose.tz;
        float t4 = ape_params.fy*r21 + ape_params.cy*r31;
        float t5 = ape_params.fy*r22 + ape_params.cy*r32;
        float t7 = ape_params.fy*pose.ty + ape_params.cy*pose.tz;
        float t8 = r31;
        float t9 = r32;
        float t11 = pose.tz;

        // reject transformations make template out of boundary
        float inv_c1z = 1 / (t8*(-ape_params.tmp_real_w) + t9*(-ape_params.tmp_real_h) + t11);
        float c1x = (t0*(-ape_params.tmp_real_w) + t1*(-ape_params.tmp_real_h) + t3) * inv_c1z;
        float c1y = (t4*(-ape_params.tmp_real_w) + t5*(-ape_params.tmp_real_h) + t7) * inv_c1z;
        float inv_c2z = 1 / (t8*(+ape_params.tmp_real_w) + t9*(-ape_params.tmp_real_h) + t11);
        float c2x = (t0*(+ape_params.tmp_real_w) + t1*(-ape_params.tmp_real_h) + t3) * inv_c2z;
        float c2y = (t4*(+ape_params.tmp_real_w) + t5*(-ape_params.tmp_real_h) + t7) * inv_c2z;
        float inv_c3z = 1 / (t8*(+ape_params.tmp_real_w) + t9*(+ape_params.tmp_real_h) + t11);
        float c3x = (t0*(+ape_params.tmp_real_w) + t1*(+ape_params.tmp_real_h) + t3) * inv_c3z;
        float c3y = (t4*(+ape_params.tmp_real_w) + t5*(+ape_params.tmp_real_h) + t7) * inv_c3z;
        float inv_c4z = 1 / (t8*(-ape_params.tmp_real_w) + t9*(+ape_params.tmp_real_h) + t11);
        float c4x = (t0*(-ape_params.tmp_real_w) + t1*(+ape_params.tmp_real_h) + t3) * inv_c4z;
        float c4y = (t4*(-ape_params.tmp_real_w) + t5*(+ape_params.tmp_real_h) + t7) * inv_c4z;
        float minx = fminf(c1x, fminf(c2x, fminf(c3x, c4x)));
        float maxx = fmaxf(c1x, fmaxf(c2x, fmaxf(c3x, c4x)));
        float miny = fminf(c1y, fminf(c2y, fminf(c3y, c4y)));
        float maxy = fmaxf(c1y, fmaxf(c2y, fmaxf(c3y, c4y)));

        // reject transformations make marker too small in screen
        float two_area = (c1x - c2x) * (c1y + c2y)
            + (c2x - c3x) * (c2y + c3y)
            + (c3x - c4x) * (c3y + c4y)
            + (c4x - c1x) * (c4y + c1y);
        float area = abs(two_area / 2);

        const int margin = 1;
        if (area > area_thres
            && (minx >= margin)
            && (maxx <= ape_params.iw - 1 - margin)
            && (miny >= margin)
            && (maxy <= ape_params.ih - 1 - margin))
            valids[i] = true;
        else
            valids[i] = false;
    }
}

__global__
void fetchTzKernel(const Pose *poses,
                   size_t num_poses,
                   float *tzs) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_poses)
        return;
    tzs[idx] = poses[idx].tz;
}

}  // namespace ape
