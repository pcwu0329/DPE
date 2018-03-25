#include <iostream>
#include <cuda.h>
#include <cublas.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <mex.h>
#include "poseRefinement.h"
#include "gradientDescentRefinement.h"
#include "prPreCal.h"
#include "prCommon.h"
#include "prUtil.h"

namespace pr {

cv::Mat poseRefinement(const cv::Mat &tmp,
                       const cv::Mat &img,
                       const std::vector<cv::Mat> &ex_mats,
                       float fx,
                       float fy,
                       float cx,
                       float cy,
                       float min_dim,
                       int prm_lvls,
                       bool photo_inva,
                       bool verbose) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::setDevice(0);
        cv::cuda::resetDevice();
        Timer time;
        long long t1;

        if (verbose) {
            time.Reset();
            time.Start();
        }

        RpParams rp_params;
        cv::cuda::GpuMat tmp_d, img_d;

        // pre-calculation
        preCal(tmp, img, ex_mats, fx, fy, cx, cy, min_dim, prm_lvls, photo_inva, &tmp_d, &img_d, &rp_params);

        if (verbose) {
            time.Pause();
            t1 = time.GetCount();
            time.Reset();
            time.Start();
        }
        // gradient descent refinement
        cv::Mat ex_mat = GradientDescentRefinement(tmp_d,
                                                   img_d,
                                                   rp_params,
                                                   ex_mats,
                                                   verbose);
        if (verbose) {
            time.Pause();
            mexPrintf("[*** Pose Refinement ***] Runtime: %f seconds\n", (t1+time.GetCount()) / 1e6);
            mexEvalString("drawnow;");
        }
        return ex_mat;
    }
    else
        return (cv::Mat_<float>(4, 4) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

void OutputExMat(const cv::Mat ex_mat,
                 std::ostream &fout) {
    fout << std::setprecision(10) << ex_mat.at<float>(0,0) << " "
         << std::setprecision(10) << ex_mat.at<float>(1,0) << " "
         << std::setprecision(10) << ex_mat.at<float>(2,0) << " "
         << std::setprecision(10) << ex_mat.at<float>(0,1) << " "
         << std::setprecision(10) << ex_mat.at<float>(1,1) << " "
         << std::setprecision(10) << ex_mat.at<float>(2,1) << " "
         << std::setprecision(10) << ex_mat.at<float>(0,2) << " "
         << std::setprecision(10) << ex_mat.at<float>(1,2) << " "
         << std::setprecision(10) << ex_mat.at<float>(2,2) << " "
         << std::setprecision(10) << ex_mat.at<float>(0,3) << " "
         << std::setprecision(10) << ex_mat.at<float>(1,3) << " "
         << std::setprecision(10) << ex_mat.at<float>(2,3)  << std::endl;
}

}  // namespace pr
