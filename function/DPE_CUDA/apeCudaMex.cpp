#include <opencv2/opencv.hpp>
#include <mex.h>
#include <opencvmex.hpp>
#include "approxPoseEsti.h"

using namespace ape;

// Approximate Pose Estimation
//
// Usage:
//   ex_mat = apeCudaMex(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, ...
//                       epsilon, prm_lvls, photo_inva, verbose);                 
//
// Inputs:
//   tmp        = template image (uint8)
//   img        = camera frame (uint8)
//   fx         = focal length along the X-axis
//   fy         = focal length along the Y-axis
//   cx         = x coordinate of the camera's principle point 
//   cy         = y coordinate of the camera's principle point 
//   min_dim    = half the length of the shorter side (real dimension)
//   min_tz     = lower bound of translation z
//   max_tz     = upper bound of translation z
//   epsilon    = delone set parameter
//   prm_lvls   = pyramid levels
//   photo_inva = need to be photometric invariant
//   verbose    = show the state of the method
//
// Outputs:
//   ex_mat = estimated extrinsic matrix (4*4)

void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[]) {
    // get input data
    cv::Mat tmp, img;
    ocvMxArrayToImage_uint8(prhs[0], tmp);
    ocvMxArrayToImage_uint8(prhs[1], img);
    float fx = float(mxGetScalar(prhs[2]));
    float fy = float(mxGetScalar(prhs[3]));
    float cx = float(mxGetScalar(prhs[4]));
    float cy = float(mxGetScalar(prhs[5]));
    float min_dim = float(mxGetScalar(prhs[6]));
    float min_tz = float(mxGetScalar(prhs[7]));
    float max_tz = float(mxGetScalar(prhs[8]));
    float epsilon = float(mxGetScalar(prhs[9]));
    int prm_lvls = int(mxGetScalar(prhs[10]));
    bool photo_inva = mxIsLogicalScalarTrue(prhs[11]);
    bool verbose = mxIsLogicalScalarTrue(prhs[12]);
    
    // mxArray for outputs
    plhs[0] = mxCreateDoubleMatrix(4, 4, mxREAL);    // 16 for 4*4 matrix
    double *ex_mat = mxGetPr(plhs[0]);
    
    approxPoseEsti(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, 
                   epsilon, prm_lvls, photo_inva, verbose, ex_mat);
}
