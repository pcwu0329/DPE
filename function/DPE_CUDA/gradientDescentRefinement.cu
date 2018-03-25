#define NOMINMAX
#include <cmath>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <cuda_texture_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <texture_fetch_functions.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <mex.h>
#include "gradientDescentRefinement.h"
#include "prCommon.h"
#include "prUtil.h"

namespace pr {

static const int BLOCK_SIZE_1D = 256;
__device__ const float Y_COEF = 0.5f;
__device__ const float CB_COEF = 0.25f;
__device__ const float CR_COEF = 0.25f;

// texture memory
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_img;
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_img_u;
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_img_v;
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_tmp;

cv::Mat GradientDescentRefinement(const cv::cuda::GpuMat &tmp,
                                  const cv::cuda::GpuMat &img,
                                  const RpParams &rp_params,
                                  const std::vector<cv::Mat> &ex_mats,
                                  bool verbose) {
    cudaChannelFormatDesc cuda_channel_format_desc = cudaCreateChannelDesc<float4>();

    // bind texture memory (template image)
    tex_tmp.addressMode[0] = cudaAddressModeBorder;
    tex_tmp.addressMode[1] = cudaAddressModeBorder;
    tex_tmp.filterMode = cudaFilterModeLinear;
    tex_tmp.normalized = false;

    // bind texture memory (camera frame)
    tex_img.addressMode[0] = cudaAddressModeBorder;
    tex_img.addressMode[1] = cudaAddressModeBorder;
    tex_img.filterMode = cudaFilterModeLinear;
    tex_img.normalized = false;

    // bind texture memory (camera frame)
    tex_img_u.addressMode[0] = cudaAddressModeBorder;
    tex_img_u.addressMode[1] = cudaAddressModeBorder;
    tex_img_u.filterMode = cudaFilterModeLinear;
    tex_img_u.normalized = false;

    // bind texture memory (camera frame)
    tex_img_v.addressMode[0] = cudaAddressModeBorder;
    tex_img_v.addressMode[1] = cudaAddressModeBorder;
    tex_img_v.filterMode = cudaFilterModeLinear;
    tex_img_v.normalized = false;

    cv::Ptr<cv::cuda::Filter> sobel_filter_x = cv::cuda::createSobelFilter(CV_32FC4, CV_32FC4, 1, 0, 3, 0.125);
    cv::Ptr<cv::cuda::Filter> sobel_filter_y = cv::cuda::createSobelFilter(CV_32FC4, CV_32FC4, 0, 1, 3, 0.125);

    int pose_num = ex_mats.size();

    // fetch parameters
    cv::Mat in_mat = rp_params.in_mat.clone();
    cv::Mat nm_mat = rp_params.nm_mat.clone();
    int iw = rp_params.iw;
    int ih = rp_params.ih;
    int tw = rp_params.tw;
    int th = rp_params.th;
    float tmp_real_w = rp_params.tmp_real_w;
    float tmp_real_h = rp_params.tmp_real_h;

    // pyramidal pose refinement
    float scale = 1.f / powf(2, rp_params.prm_lvls - 1);
    int max_iter = 10;
    cv::Mat small_in_mat = in_mat.clone();
    float epsilon_r = 1.f / scale / scale / 256;
    float epsilon_t = 1.f / scale / scale / 256;
    float offset = 0.5f;
    auto temp_ex_mats = ex_mats;
    std::vector<float> apr_errs(pose_num, 1);

    Timer time;
    time.Reset();
    time.Start();

    while (scale <= 1.f / 2.f) {
        cv::cuda::GpuMat small_img, small_img_u, small_img_v;
        cv::cuda::resize(img, small_img, cv::Size(), scale, scale, CV_INTER_AREA);

        // ((cv::Mat)(temp_mat1 + offset)*scale - offset) is too slow; compute the values separately
        small_in_mat.at<float>(0, 0) = in_mat.at<float>(0, 0) * scale;
        small_in_mat.at<float>(1, 1) = in_mat.at<float>(1, 1) * scale;
        small_in_mat.at<float>(0, 2) = (in_mat.at<float>(0, 2) + offset) * scale - offset;
        small_in_mat.at<float>(1, 2) = (in_mat.at<float>(1, 2) + offset) * scale - offset;
        int small_iw = small_img.cols;
        int small_ih = small_img.rows;
        sobel_filter_x->apply(small_img, small_img_u);
        sobel_filter_y->apply(small_img, small_img_v);
        cudaBindTexture2D(0, &tex_img, small_img.data, &cuda_channel_format_desc, small_iw, small_ih, small_img.step);
        cudaBindTexture2D(0, &tex_img_u, small_img_u.data, &cuda_channel_format_desc, small_iw, small_ih, small_img_u.step);
        cudaBindTexture2D(0, &tex_img_v, small_img_v.data, &cuda_channel_format_desc, small_iw, small_ih, small_img_v.step);

        for (int i = 0; i < pose_num; ++i) {
            if (apr_errs[i] != 2) {
                thrust::device_vector<float4> tmp_vals;
                thrust::device_vector<float2> tmp_coors;
                bool validness = CalValidCoors(tmp,
                                               small_in_mat,
                                               temp_ex_mats[i],
                                               nm_mat,
                                               tw,
                                               th,
                                               small_iw,
                                               small_ih,
                                               &tmp_vals,
                                               &tmp_coors);
                if (validness)
                    apr_errs[i] = Gdr(tmp_vals,
                                      tmp_coors,
                                      small_in_mat,
                                      make_int2(small_iw, small_ih),
                                      make_float2(tmp_real_w, tmp_real_h),
                                      epsilon_r,
                                      epsilon_t,
                                      max_iter,
                                      scale,
                                      verbose,
                                      temp_ex_mats[i]);
                else
                    apr_errs[i] = 2;
            }
        }
        cudaUnbindTexture(&tex_img);
        cudaUnbindTexture(&tex_img_u);
        cudaUnbindTexture(&tex_img_v);
        scale *= 2;
        epsilon_r /= 4;
        epsilon_t /= 4;
    }

    time.Start();
    cv::Mat ex_mat;

    // get the one with smaller appearance error
    if (std::accumulate(apr_errs.begin(), apr_errs.end(), 0) == pose_num * 2) {
        return ex_mats[0];
    }
    else if (pose_num == 1 || apr_errs[0] <= apr_errs[1]) {
        ex_mat = temp_ex_mats[0];
    }
    else {
        if (verbose) {
            mexPrintf("Switch to the other pose!\n");
            mexEvalString("drawnow;");
        }
        ex_mat = temp_ex_mats[1];
    }

    // final refinement
    thrust::device_vector<float4> tmp_vals;
    thrust::device_vector<float2> tmp_coors;
    cv::cuda::GpuMat img_u, img_v;
    sobel_filter_x->apply(img, img_u);
    sobel_filter_y->apply(img, img_v); 
    cudaBindTexture2D(0, &tex_img, img.data, &cuda_channel_format_desc, iw, ih, img.step);
    cudaBindTexture2D(0, &tex_img_u, img_u.data, &cuda_channel_format_desc, iw, ih, img_u.step);
    cudaBindTexture2D(0, &tex_img_v, img_v.data, &cuda_channel_format_desc, iw, ih, img_v.step);
    cudaThreadSynchronize();

    bool validness = CalValidCoors(tmp,
                                   in_mat,
                                   ex_mat,
                                   nm_mat,
                                   tw,
                                   th,
                                   iw,
                                   ih,
                                   &tmp_vals,
                                   &tmp_coors);
    if (validness) {
        Gdr(tmp_vals,
            tmp_coors,
            in_mat,
            make_int2(iw, ih),
            make_float2(tmp_real_w, tmp_real_h),
            epsilon_r,
            epsilon_t,
            max_iter,
            1,
            verbose,
            ex_mat);
    }
    else
        ex_mat = ex_mats[0];

    cudaUnbindTexture(&tex_img);
    cudaUnbindTexture(&tex_img_u);
    cudaUnbindTexture(&tex_img_v);

    return ex_mat;
}

bool CalValidCoors(const cv::cuda::GpuMat &tmp,
                   const cv::Mat &in_mat,
                   const cv::Mat &ex_mat,
                   const cv::Mat &nm_mat,
                   int tw,
                   int th,
                   int iw,
                   int ih,
                   thrust::device_vector<float4> *tmp_vals,
                   thrust::device_vector<float2> *tmp_coors) {
    cv::Mat boundary = (cv::Mat_<float>(3, 4) << 0, 0, tw - 1, tw - 1, 0, th - 1, th - 1, 0, 1, 1, 1, 1);
    cv::Mat H = getHomoMatFromInExNm(in_mat, ex_mat, nm_mat);
    Region region;
    cv::Mat corners;
    calRegion(boundary,
              H,
              tw,
              th,
              iw,
              ih,
              1,
              &corners,
              &region);
    int crop_w = region.x_max - region.x_min + 1;
    int crop_h = region.y_max - region.y_min + 1;
    int4 rect = make_int4(region.x_min, region.y_min, crop_w, crop_h);
    int num = crop_w * crop_h;
    tmp_vals->resize(num);
    tmp_coors->resize(num);
    thrust::device_vector<bool> valids(num, true);
    cv::Mat inv_H = H.inv();
    inv_H /= inv_H.at<float>(2, 2);
    int2 tmp_size = make_int2(tw, th);
    float4 nm_params = make_float4(nm_mat.at<float>(0,0), nm_mat.at<float>(1, 1), nm_mat.at<float>(0, 2), nm_mat.at<float>(1, 2));
    float4 homo_1st = make_float4(inv_H.at<float>(0, 0), inv_H.at<float>(1, 0), inv_H.at<float>(2, 0), inv_H.at<float>(0, 1));
    float4 homo_2nd = make_float4(inv_H.at<float>(1, 1), inv_H.at<float>(2, 1), inv_H.at<float>(0, 2), inv_H.at<float>(1, 2));
    float area = polygonArea(corners(cv::Rect(0, 0, 4, 2)));
    if (area < 16) return false;
    float area_ori = polygonArea(boundary(cv::Rect(0, 0, 4, 2)));
    float scale = sqrt(area / area_ori);

    cv::cuda::GpuMat temp_tmp;
    // Resize the template image so that it would have the similar size with the projected one
    // It only affects the pixel fetching process. The coordinate computation keeps the same.
    if (scale < 1) {
        cv::cuda::resize(tmp, temp_tmp, cv::Size(), scale, scale, CV_INTER_AREA);
        cv::cuda::resize(temp_tmp, temp_tmp, cv::Size(tw, th));
    }
    else {
        cv::cuda::resize(tmp, temp_tmp, cv::Size(), scale, scale, CV_INTER_LINEAR);
        cv::cuda::resize(temp_tmp, temp_tmp, cv::Size(tw, th), 0, 0, CV_INTER_AREA);
    }
    cudaChannelFormatDesc cuda_channel_format_desc = cudaCreateChannelDesc<float4>();
    cudaBindTexture2D(0, &tex_tmp, temp_tmp.data, &cuda_channel_format_desc, tw, th, temp_tmp.step);
    const int BLOCK_NUM = (num - 1) / BLOCK_SIZE_1D + 1;

    CalValidCoorsKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(rect,
                                                      num,
                                                      tmp_size,
                                                      nm_params,
                                                      homo_1st,
                                                      homo_2nd,
                                                      thrust::raw_pointer_cast(tmp_vals->data()),
                                                      thrust::raw_pointer_cast(tmp_coors->data()),
                                                      thrust::raw_pointer_cast(valids.data()));

    auto zip_it_valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(tmp_vals->begin(), tmp_coors->begin(), valids.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(tmp_vals->end(), tmp_coors->end(), valids.end())),
        ValidTmpCoorFunctor());
    tmp_vals->erase(thrust::get<0>(zip_it_valid_end.get_iterator_tuple()), tmp_vals->end());
    tmp_coors->erase(thrust::get<1>(zip_it_valid_end.get_iterator_tuple()), tmp_coors->end());
    cudaUnbindTexture(&tex_tmp);
    if (tmp_vals->size() < 16)
        return false;
    else
        return true;
}

__global__
void CalValidCoorsKernel(int4 rect,
                         int num,
                         int2 tmp_size,
                         float4 nm_params,
                         float4 homo_1st,
                         float4 homo_2nd,
                         float4 *tmp_vals,
                         float2 *tmp_coors,
                         bool *valids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num)
        return;

    int u = (idx % rect.z) + rect.x;
    int v = (idx / rect.z) + rect.y;

    float inv_z = 1.f / (homo_1st.z*u + homo_2nd.y*v + 1);
    float x = (homo_1st.x*u + homo_1st.w*v + homo_2nd.z) * inv_z;
    float y = (homo_1st.y*u + homo_2nd.x*v + homo_2nd.w) * inv_z;
    if (x < 0 || x > tmp_size.x - 1.f  || y < 0 || y > tmp_size.y - 1.f) {
        valids[idx] = false;
        return;
    }
    // We resize the template and it affects the pixel fetching process
    tmp_vals[idx] = tex2D(tex_tmp, x+0.5f, y+0.5f);
    tmp_coors[idx] = make_float2(nm_params.x*x + nm_params.z, nm_params.y*y + nm_params.w);
}

// return appearance error
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
          cv::Mat &ex_mat) {

    // check if the template is within the image
    auto H = getHomoMatFromInEx(in_mat, ex_mat);
    float4 homo_1st = make_float4(H.at<float>(0, 0), H.at<float>(1, 0), H.at<float>(2, 0), H.at<float>(0, 1));
    float4 homo_2nd = make_float4(H.at<float>(1, 1), H.at<float>(2, 1), H.at<float>(0, 2), H.at<float>(1, 2));
    if (!WithinRegion(img_size, tmp_real, homo_1st, homo_2nd))
        return 1;

    auto p = getAaParametersFromExtrinsicMatirx(ex_mat);
    auto ori_p = p.clone();
    float diff_r = 1e10f;
    float diff_t = 1e10f;
    int num = tmp_vals.size();

    float* apr_errs;
    cudaMalloc((void **)&apr_errs, num * 3);
    thrust::device_vector<float4> img_u_vals(num);
    thrust::device_vector<float4> img_v_vals(num);
    CalAprErrAndGradient(tmp_vals,
                         tmp_coors,
                         num,
                         in_mat,
                         ex_mat,
                         img_size,
                         apr_errs,
                         &img_u_vals,
                         &img_v_vals);
    cudaDeviceSynchronize();
    float apr_err = cublasSasum(num*3, apr_errs, 1) / num;

    if (verbose) {
        mexPrintf("Initial Condition: epsilon_r = %.6f, epsilon_t = %.6f, Ea = %.6f\n", epsilon_r, epsilon_t, apr_err);
        mexEvalString("drawnow;");
    }

    // prepare data
    const int BLOCK_NUM = (num - 1) / BLOCK_SIZE_1D + 1;
    const int rows = num * 3;
    float *J, *JtJ_gpu, *JtE_gpu, *JtJ_cpu, *JtE_cpu;
    cudaMalloc((void **)&J, rows * 6 * sizeof(float));
    cudaMalloc((void **)&JtJ_gpu, 6 * 6 * sizeof(float));
    cudaMalloc((void **)&JtE_gpu, 6 * 1 * sizeof(float));
    JtJ_cpu = new float[6 * 6];
    JtE_cpu = new float[6 * 1];
    cv::Mat JtE(6, 1, CV_32F);
    cv::Mat delta_p(6, 1, CV_32F);
    float2 in_mat_params = make_float2(in_mat.at<float>(0, 0), in_mat.at<float>(1, 1));

    // if the current camera frame is a smaller one, then we can allow more iterations
    max_iter /= scale;
    int iter = 0;
    while ((diff_r > epsilon_r || diff_t > epsilon_t) && iter < max_iter) {
        // --- Step 0: Prepare necessary matrices ---
        ex_mat = getExtrinsicMatrixFromAaParameters(p);
        auto H = getHomoMatFromInEx(in_mat, ex_mat);
        float4 homo_1st = make_float4(H.at<float>(0, 0), H.at<float>(1, 0), H.at<float>(2, 0), H.at<float>(0, 1));
        float4 homo_2nd = make_float4(H.at<float>(1, 1), H.at<float>(2, 1), H.at<float>(0, 2), H.at<float>(1, 2));
        if (!WithinRegion(img_size, tmp_real, homo_1st, homo_2nd)) break;
        R12t R12_t = getR12t(ex_mat);
        JacobRr J_Rr = getJacobRr(p(cv::Rect(0, 0, 1, 3)));

        // --- Step 1: Compute Jacobian ---
        CalJacobianKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(thrust::raw_pointer_cast(tmp_coors.data()),
                                                        thrust::raw_pointer_cast(img_u_vals.data()),
                                                        thrust::raw_pointer_cast(img_v_vals.data()),
                                                        num,
                                                        in_mat_params,
                                                        R12_t,
                                                        J_Rr,
                                                        J);
        cudaDeviceSynchronize();

        // --- Step 2: Compute JtJ_cpu ---
        cublasSgemm('t', 'n', 6, 6, rows, 1, J, rows, J, rows, 0, JtJ_gpu, 6);

        // --- Step 3: Compute delta p
        cublasSgemm('t', 'n', 6, 1, rows, 1, J, rows, apr_errs, rows, 0, JtE_gpu, 6);
        cudaMemcpy(JtJ_cpu, JtJ_gpu, 36 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(JtE_cpu, JtE_gpu, 6 * sizeof(float), cudaMemcpyDeviceToHost);
        calDeltaP(JtJ_cpu,
                  JtE_cpu,
                  &delta_p,
                  &JtE);

        // --- Step 4: Backtracking line search
        const float alpha = 0.5f;
        const float c = 1e-4f;
        CalAprErrAndGradient(tmp_vals,
                             tmp_coors,
                             num,
                             in_mat,
                             getExtrinsicMatrixFromAaParameters(p + delta_p),
                             img_size,
                             apr_errs,
                             &img_u_vals,
                             &img_v_vals);
        float new_apr_err = cublasSasum(num * 3, apr_errs, 1) / num;
        calPoseDiff(p, p + delta_p, &diff_r, &diff_t);
        while ((new_apr_err > apr_err + (-2.0 * c / num * delta_p.dot(JtE))) &&
               (diff_r > epsilon_r || diff_t > epsilon_t)) {
            delta_p = delta_p * alpha;
            CalAprErrAndGradient(tmp_vals,
                                 tmp_coors,
                                 num,
                                 in_mat,
                                 getExtrinsicMatrixFromAaParameters(p + delta_p),
                                 img_size,
                                 apr_errs,
                                 &img_u_vals,
                                 &img_v_vals);
            new_apr_err = cublasSasum(num * 3, apr_errs, 1) / num;
            calPoseDiff(p, p + delta_p, &diff_r, &diff_t);
        }
        apr_err = new_apr_err;

        // --- Step 5: Update the parameters ---
        p = p + delta_p;
        iter = iter + 1;

        if (verbose) {
            mexPrintf("Iteration: %3d, Ea = %.6f, diff_r = %.6f, diff_t = %.6f\n", iter, apr_err, diff_r, diff_t);
            mexEvalString("drawnow;");
        }
    }
    ex_mat = getExtrinsicMatrixFromAaParameters(p);

    cudaFree(apr_errs);
    cudaFree(J);
    cudaFree(JtJ_gpu);
    cudaFree(JtE_gpu);
    delete[] JtJ_cpu;
    delete[] JtE_cpu;
    return apr_err;
}

void CalAprErrAndGradient(const thrust::device_vector<float4> &tmp_vals,
                          const thrust::device_vector<float2> &tmp_coors,
                          int num,
                          const cv::Mat &in_mat,
                          const cv::Mat &ex_mat,
                          int2 img_size,
                          float *apr_errs,
                          thrust::device_vector<float4> *img_u_vals,
                          thrust::device_vector<float4> *img_v_vals) {
    auto H = getHomoMatFromInEx(in_mat, ex_mat);
    float4 homo_1st = make_float4(H.at<float>(0, 0), H.at<float>(1, 0), H.at<float>(2, 0), H.at<float>(0, 1));
    float4 homo_2nd = make_float4(H.at<float>(1, 1), H.at<float>(2, 1), H.at<float>(0, 2), H.at<float>(1, 2));
    const int BLOCK_NUM = (num - 1) / BLOCK_SIZE_1D + 1;
    CalAprErrAndGradientKernel<<<BLOCK_NUM, BLOCK_SIZE_1D>>>(thrust::raw_pointer_cast(tmp_vals.data()),
                                                             thrust::raw_pointer_cast(tmp_coors.data()),
                                                             num,
                                                             img_size,
                                                             homo_1st,
                                                             homo_2nd,
                                                             apr_errs,
                                                             thrust::raw_pointer_cast(img_u_vals->data()),
                                                             thrust::raw_pointer_cast(img_v_vals->data()));
}

__global__
void CalAprErrAndGradientKernel(const float4 *tmp_vals,
                                const float2 *tmp_coors,
                                int num,
                                int2 img_size,
                                float4 homo_1st,
                                float4 homo_2nd,
                                float *apr_errs,
                                float4 *img_u_vals,
                                float4 *img_v_vals) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num)
        return;

    float x = tmp_coors[idx].x;
    float y = tmp_coors[idx].y;

    float inv_h = 1.f / (homo_1st.z*x + homo_2nd.y*y + 1);
    // have to add 0.5f for coordinates (see E.2 Linear Filtering in CUDA Programming Guide)
    float u = (homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_h + 0.5f;
    float v = (homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_h + 0.5f;

    float4 tmp_val = tmp_vals[idx];
    
    float4 img_val = tex2D(tex_img, u, v);
    int cur_idx = idx;
    apr_errs[cur_idx] = Y_COEF * (tmp_val.x - img_val.x);
    cur_idx += num;
    apr_errs[cur_idx] = CB_COEF * (tmp_val.y - img_val.y);
    cur_idx += num;
    apr_errs[cur_idx] = CR_COEF * (tmp_val.z - img_val.z);

    if (img_u_vals)
        img_u_vals[idx] = tex2D(tex_img_u, u, v);

    if (img_v_vals)
        img_v_vals[idx] = tex2D(tex_img_v, u, v);
}

bool WithinRegion(int2 img_size,
                  float2 tmp_real,
                  float4 homo_1st,
                  float4 homo_2nd) {
    float inv_h, u, v, x, y;
    x = tmp_real.x;
    y = tmp_real.y;
    inv_h = 1.f / (homo_1st.z*x + homo_2nd.y*y + 1);
    u = (homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_h;
    v = (homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_h;
    if (u < 0 || u > img_size.x-1 || v < 0 || v > img_size.y-1)
        return false;

    x = -tmp_real.x;
    y = -tmp_real.y;
    inv_h = 1.f / (homo_1st.z*x + homo_2nd.y*y + 1);
    u = (homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_h;
    v = (homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_h;
    if (u < 0 || u > img_size.x - 1 || v < 0 || v > img_size.y - 1)
        return false;

    x = tmp_real.x;
    y = -tmp_real.y;
    inv_h = 1.f / (homo_1st.z*x + homo_2nd.y*y + 1);
    u = (homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_h;
    v = (homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_h;
    if (u < 0 || u > img_size.x - 1 || v < 0 || v > img_size.y - 1)
        return false;

    x = -tmp_real.x;
    y = tmp_real.y;
    inv_h = 1.f / (homo_1st.z*x + homo_2nd.y*y + 1);
    u = (homo_1st.x*x + homo_1st.w*y + homo_2nd.z) * inv_h;
    v = (homo_1st.y*x + homo_2nd.x*y + homo_2nd.w) * inv_h;
    if (u < 0 || u > img_size.x - 1 || v < 0 || v > img_size.y - 1)
        return false;

    return true;
}

__global__
void CalJacobianKernel(const float2 *tmp_coors,
                       const float4 *img_u_vals,
                       const float4 *img_v_vals,
                       int num,
                       float2 in_mat_params,
                       R12t R12_t,
                       JacobRr J_Rr,
                       float *J)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num)
        return;

    const float2 &tmp_coor = tmp_coors[idx];
    const float4 &img_u_val = img_u_vals[idx];
    const float4 &img_v_val = img_v_vals[idx];

    const float &fx = in_mat_params.x;
    const float &fy = in_mat_params.y;
    const float &x = tmp_coor.x;
    const float &y = tmp_coor.y;
    const int rows = num * 3;
    float xc = R12_t.r11 * x + R12_t.r12 * y + R12_t.t1;
    float yc = R12_t.r21 * x + R12_t.r22 * y + R12_t.t2;
    float izc = 1.f / (R12_t.r31 * x + R12_t.r32 * y + R12_t.t3);
    float izc2 = izc * izc;
    float fx_iz = fx * izc;
    float fy_iz = fy * izc;
    float fxxc_zc2_ = -fx * xc * izc2;
    float fyyc_zc2_ = -fy * yc * izc2;

    // Y channel
    int cur_idx = idx;
    float fxYu_zc = fx_iz * img_u_val.x;
    float fyYv_zc = fy_iz * img_v_val.x;
    float fxxcYu_zc2_ = fxxc_zc2_ * img_u_val.x;
    float fyycYv_zc2_ = fyyc_zc2_ * img_v_val.x;
    float j1 = fxYu_zc*x;
    float j2 = fxYu_zc*y;
    float j3 = fyYv_zc*x;
    float j4 = fyYv_zc*y;
    float j5 = fxxcYu_zc2_*x + fyycYv_zc2_*x;
    float j6 = fxxcYu_zc2_*y + fyycYv_zc2_*y;
    int offset = 0;
    J[cur_idx] = Y_COEF * (j1 * J_Rr.j11 + j2 * J_Rr.j21 + j3 * J_Rr.j31 + j4 * J_Rr.j41 + j5 * J_Rr.j51 + j6 * J_Rr.j61);
    J[cur_idx + (offset += rows)] = Y_COEF * (j1 * J_Rr.j12 + j2 * J_Rr.j22 + j3 * J_Rr.j32 + j4 * J_Rr.j42 + j5 * J_Rr.j52 + j6 * J_Rr.j62);
    J[cur_idx + (offset += rows)] = Y_COEF * (j1 * J_Rr.j13 + j2 * J_Rr.j23 + j3 * J_Rr.j33 + j4 * J_Rr.j43 + j5 * J_Rr.j53 + j6 * J_Rr.j63);
    J[cur_idx + (offset += rows)] = Y_COEF * (fxYu_zc);
    J[cur_idx + (offset += rows)] = Y_COEF * (fyYv_zc);
    J[cur_idx + (offset += rows)] = Y_COEF * (fxxcYu_zc2_ + fyycYv_zc2_);

    // Cb channel
    cur_idx += num;
    float fxCbu_zc = fx_iz * img_u_val.y;
    float fyCbv_zc = fy_iz * img_v_val.y;
    float fxxcCbu_zc2_ = fxxc_zc2_ * img_u_val.y;
    float fyycCbv_zc2_ = fyyc_zc2_ * img_v_val.y;
    j1 = fxCbu_zc*x;
    j2 = fxCbu_zc*y;
    j3 = fyCbv_zc*x;
    j4 = fyCbv_zc*y;
    j5 = fxxcCbu_zc2_*x + fyycCbv_zc2_*x;
    j6 = fxxcCbu_zc2_*y + fyycCbv_zc2_*y;
    offset = 0;
    J[cur_idx] = CB_COEF * (j1 * J_Rr.j11 + j2 * J_Rr.j21 + j3 * J_Rr.j31 + j4 * J_Rr.j41 + j5 * J_Rr.j51 + j6 * J_Rr.j61);
    J[cur_idx + (offset += rows)] = CB_COEF * (j1 * J_Rr.j12 + j2 * J_Rr.j22 + j3 * J_Rr.j32 + j4 * J_Rr.j42 + j5 * J_Rr.j52 + j6 * J_Rr.j62);
    J[cur_idx + (offset += rows)] = CB_COEF * (j1 * J_Rr.j13 + j2 * J_Rr.j23 + j3 * J_Rr.j33 + j4 * J_Rr.j43 + j5 * J_Rr.j53 + j6 * J_Rr.j63);
    J[cur_idx + (offset += rows)] = CB_COEF * (fxCbu_zc);
    J[cur_idx + (offset += rows)] = CB_COEF * (fyCbv_zc);
    J[cur_idx + (offset += rows)] = CB_COEF * (fxxcCbu_zc2_ + fyycCbv_zc2_);

    // Cr channel
    cur_idx += num;
    float fxCru_zc = fx_iz * img_u_val.z;
    float fyCrv_zc = fy_iz * img_v_val.z;
    float fxxcCru_zc2_ = fxxc_zc2_ * img_u_val.z;
    float fyycCrv_zc2_ = fyyc_zc2_ * img_v_val.z;
    j1 = fxCru_zc*x;
    j2 = fxCru_zc*y;
    j3 = fyCrv_zc*x;
    j4 = fyCrv_zc*y;
    j5 = fxxcCru_zc2_*x + fyycCrv_zc2_*x;
    j6 = fxxcCru_zc2_*y + fyycCrv_zc2_*y;
    offset = 0;
    J[cur_idx] = CR_COEF * (j1 * J_Rr.j11 + j2 * J_Rr.j21 + j3 * J_Rr.j31 + j4 * J_Rr.j41 + j5 * J_Rr.j51 + j6 * J_Rr.j61);
    J[cur_idx + (offset += rows)] = CR_COEF * (j1 * J_Rr.j12 + j2 * J_Rr.j22 + j3 * J_Rr.j32 + j4 * J_Rr.j42 + j5 * J_Rr.j52 + j6 * J_Rr.j62);
    J[cur_idx + (offset += rows)] = CR_COEF * (j1 * J_Rr.j13 + j2 * J_Rr.j23 + j3 * J_Rr.j33 + j4 * J_Rr.j43 + j5 * J_Rr.j53 + j6 * J_Rr.j63);
    J[cur_idx + (offset += rows)] = CR_COEF * (fxCru_zc);
    J[cur_idx + (offset += rows)] = CR_COEF * (fyCrv_zc);
    J[cur_idx + (offset += rows)] = CR_COEF * (fxxcCru_zc2_ + fyycCrv_zc2_);
}

}  // namespace pr
