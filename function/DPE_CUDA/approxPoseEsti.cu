#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <mex.h>
#include "approxPoseEsti.h"
#include "coarseToFinePoseEsti.h"
#include "apePreCal.h"
#include "apeCommon.h"

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
                    double *ex_mat) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::setDevice(0);
        cv::cuda::resetDevice();

        Timer time;
        long long t1;

        if (verbose) {
            time.Reset();
            time.Start();
        }
        ApeParams ape_params;

        // allocate
        cv::cuda::GpuMat tmp_d(tmp.rows, tmp.cols, CV_32FC3);
        cv::cuda::GpuMat img_d(img.rows, img.cols, CV_32FC4);
        // pre-calculation
        preCal(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, epsilon, verbose, &tmp_d, &img_d, &ape_params);

        if (verbose) {
            time.Pause();
            t1 = time.GetCount();
            time.Reset();
            time.Start();
        }

        // coarse-to-fine pose estimation
        coarseToFinePoseEstimation(tmp_d, img_d, prm_lvls, photo_inva, verbose, &ape_params, ex_mat);
        
        if (verbose) {
            time.Pause();
            mexPrintf("[*** Approximation Pose Estimation ***] Runtime: %f seconds\n", (t1+time.GetCount()) / 1e6);
            mexEvalString("drawnow;");
        }
    }
}

}  // namespace ape
