#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <mex.h>
#include <opencvmex.hpp>
#include "poseRefinement.h"
#include "prUtil.h"

using namespace pr;

// Pose Refinement
//
// Usage:
//   ex_mat = prCudaMex(tmp, img, ex_mats, fx, fy, cx, cy, min_dim, prm_lvls, photo_inva, verbose);                 
//
// Inputs:
//   tmp        = template image (uint8)
//   img        = camera frame (uint8)
//   ex_mats    = estimated extrinsic matrices (4*8 or 4*4)
//   fx         = focal length along the X-axis
//   fy         = focal length along the Y-axis
//   cx         = x coordinate of the camera's principle point 
//   cy         = y coordinate of the camera's principle point 
//   min_dim    = half the length of the shorter side (real dimension)
//   prm_lvls   = pyramid levels
//   photo_inva = need to be photometric invariant
//   verbose    = show the state of the method
//
// Outputs:
//   ex_mat = refined extrinsic matrix (4*4)

void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[]) {
    // get input data
    cv::Mat tmp, img;
    ocvMxArrayToImage_uint8(prhs[0], tmp);
    ocvMxArrayToImage_uint8(prhs[1], img);
    double *amb_ex_mats = mxGetPr(prhs[2]);
    float fx = float(mxGetScalar(prhs[3]));
    float fy = float(mxGetScalar(prhs[4]));
    float cx = float(mxGetScalar(prhs[5]));
    float cy = float(mxGetScalar(prhs[6]));
    float min_dim = float(mxGetScalar(prhs[7]));
    int prm_lvls = int(mxGetScalar(prhs[8]));
    bool photo_inva = mxIsLogicalScalarTrue(prhs[9]);
    bool verbose = mxIsLogicalScalarTrue(prhs[10]);
    
    // mxArray for outputs
    plhs[0] = mxCreateDoubleMatrix(4, 4, mxREAL);    // 16 for 4*4 matrix
    double *ex_mat = mxGetPr(plhs[0]);
    
    int num = mxGetN(prhs[2]) / 4;
    std::vector<cv::Mat> ex_mats(num);
    for (int i = 0, j = 0; i < num; ++i, j+=16) {
        ex_mats[i] = (cv::Mat_<float>(4, 4) << amb_ex_mats[j+0], amb_ex_mats[j+4], amb_ex_mats[j+8],  amb_ex_mats[j+12],
                                               amb_ex_mats[j+1], amb_ex_mats[j+5], amb_ex_mats[j+9],  amb_ex_mats[j+13],
                                               amb_ex_mats[j+2], amb_ex_mats[j+6], amb_ex_mats[j+10], amb_ex_mats[j+14],
                                               amb_ex_mats[j+3], amb_ex_mats[j+7], amb_ex_mats[j+11], amb_ex_mats[j+15]);
    }

    cv::Mat ex_mat_cv = poseRefinement(tmp,
                                       img,
                                       ex_mats,
                                       fx,
                                       fy,
                                       cx,
                                       cy,
                                       min_dim,
                                       prm_lvls,
                                       photo_inva,
                                       verbose);
                                   
    ex_mat[0]  = double(ex_mat_cv.at<float>(0,0));
    ex_mat[1]  = double(ex_mat_cv.at<float>(1,0));
    ex_mat[2]  = double(ex_mat_cv.at<float>(2,0));
    ex_mat[3]  = double(ex_mat_cv.at<float>(3,0));
    ex_mat[4]  = double(ex_mat_cv.at<float>(0,1));
    ex_mat[5]  = double(ex_mat_cv.at<float>(1,1));
    ex_mat[6]  = double(ex_mat_cv.at<float>(2,1));
    ex_mat[7]  = double(ex_mat_cv.at<float>(3,1));
    ex_mat[8]  = double(ex_mat_cv.at<float>(0,2));
    ex_mat[9]  = double(ex_mat_cv.at<float>(1,2));
    ex_mat[10] = double(ex_mat_cv.at<float>(2,2));
    ex_mat[11] = double(ex_mat_cv.at<float>(3,2));
    ex_mat[12] = double(ex_mat_cv.at<float>(0,3));
    ex_mat[13] = double(ex_mat_cv.at<float>(1,3));
    ex_mat[14] = double(ex_mat_cv.at<float>(2,3)); 
    ex_mat[15] = double(ex_mat_cv.at<float>(3,3)); 
}
