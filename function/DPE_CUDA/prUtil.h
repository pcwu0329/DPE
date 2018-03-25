#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <vector_types.h>
#include "prCommon.h"

namespace pr {
    
cv::Mat getHomoMatFromInExNm(const cv::Mat &in_mat,
                             const cv::Mat &ex_mat,
                             const cv::Mat &nm_mat);

cv::Mat getHomoMatFromInEx(const cv::Mat &in_mat,
                           const cv::Mat &ex_mat);

cv::Mat getAxisAngleFromRotationMatirx(const cv::Mat &R);

cv::Mat getRotationMatrixFromAxisAngle(const cv::Mat &r);

cv::Mat getAaParametersFromExtrinsicMatirx(const cv::Mat &ex_mat);

cv::Mat getExtrinsicMatrixFromAaParameters(const cv::Mat &p);

cv::Mat getCrossProductMatrix(const cv::Mat &r);

R12t getR12t(const cv::Mat &ex_mat);

JacobRr getJacobRr(const cv::Mat &r);

float polygonArea(const cv::Mat &corners);

void calRegion(const cv::Mat &boundary,
               const cv::Mat &H,
               int tw,
               int th,
               int iw,
               int ih,
               float scale,
               cv::Mat *corners,
               Region *region);

void calPoseDiff(const cv::Mat &p1,
                 const cv::Mat &p2,
                 float *diff_r,
                 float *diff_t);

void calDeltaP(float *JtJ_ptr,
               float *JtE_ptr,
               cv::Mat *delta_p,
               cv::Mat *JtE);
}  // namespace pr
