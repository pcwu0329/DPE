#include <mex.h>
#include <cmath>
#include <algorithm>
#include <omp.h>

// From 6D pose to 16D transformation matrix. Remove indecent ones (i.e., outside the boundaries). 
//
// Usage:
//   [trans_mats, insiders] = poseToTransMatMex(poses, ih, iw, fx, fy, cx, cy, tmp_real_w, tmp_real_h, area_thres)                  
//
// Inputs:
//   poses      = all candidate poses (6*n)
//   ih         = upper bound of rz
//   iw         = lower bound of rx
//   fx         = focal length along the X-axis
//   fy         = focal length along the Y-axis
//   cx         = x coordinate of the camera's principle point 
//   cy         = y coordinate of the camera's principle point 
//   tmp_real_w = template width in real unit
//   tmp_real_h = template height in real unit
//   area_thres = area threshold
//
// Outputs:
//   trans_mats = all transformation matrices (16*n)
//   insiders   = indication of insiders  (1*n)
                                                
void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[]) {

    int num_poses = mxGetN(prhs[0]);

    // mxArray for outputs
    plhs[0] = mxCreateDoubleMatrix(16, num_poses, mxREAL);    // 16 for 4*4 matrix
    plhs[1] = mxCreateNumericMatrix(1, num_poses, mxINT32_CLASS, mxREAL);
    double *trans = mxGetPr(plhs[0]);
    int *insiders = (int*)mxGetPr(plhs[1]);

    // get input data
    double *poses = mxGetPr(prhs[0]);
    int ih = int(mxGetScalar(prhs[1]));
    int iw = int(mxGetScalar(prhs[2]));
    double fx = mxGetScalar(prhs[3]);
    double fy = mxGetScalar(prhs[4]);
    double cx = mxGetScalar(prhs[5]);
    double cy = mxGetScalar(prhs[6]);
    double tmp_real_w = mxGetScalar(prhs[7]);
    double tmp_real_h = mxGetScalar(prhs[8]);
    double area_thres = mxGetScalar(prhs[9]);

    // main loop
    int i;
#pragma omp parallel for private(i) num_threads(8)
    for (i = 0; i < num_poses; i++) {
        // pose values
        double rz0 = poses[6 * i];
        double rx  = poses[6 * i + 1] + 3.1415926;
        double rz1 = poses[6 * i + 2];
        double tx  = poses[6 * i + 3];
        double ty  = poses[6 * i + 4];
        double tz  = poses[6 * i + 5];
        
		// calculate rotation matrix
		double r11 =  cos(rz0)*cos(rz1) - sin(rz0)*cos(rx)*sin(rz1);
		double r12 = -cos(rz0)*sin(rz1) - sin(rz0)*cos(rx)*cos(rz1);
		double r13 =  sin(rz0)*sin(rx);
		double r21 =  sin(rz0)*cos(rz1) + cos(rz0)*cos(rx)*sin(rz1);
		double r22 = -sin(rz0)*sin(rz1) + cos(rz0)*cos(rx)*cos(rz1);
		double r23 = -cos(rz0)*sin(rx);
		double r31 =  sin(rx)*sin(rz1);
		double r32 =  sin(rx)*cos(rz1);
		double r33 =  cos(rx);

        // final transfomration
        trans[16 * i]      = fx*r11 + cx*r31;
        trans[16 * i + 1]  = fx*r12 + cx*r32;
        trans[16 * i + 2]  = fx*r13 + cx*r33;
        trans[16 * i + 3]  = fx*tx  + cx*tz;
        trans[16 * i + 4]  = fy*r21 + cy*r31;
        trans[16 * i + 5]  = fy*r22 + cy*r32;
        trans[16 * i + 6]  = fy*r23 + cy*r33;
        trans[16 * i + 7]  = fy*ty  + cy*tz;
        trans[16 * i + 8]  = r31;
        trans[16 * i + 9]  = r32;
        trans[16 * i + 10] = r33;
        trans[16 * i + 11] = tz;
        trans[16 * i + 12] = 0;
        trans[16 * i + 13] = 0;
        trans[16 * i + 14] = 0;
        trans[16 * i + 15] = 1;

        // reject transformations make marker out of boundary
        double c1x = (trans[16 * i + 3]  - trans[16 * i]     * tmp_real_w - trans[16 * i + 1] * tmp_real_h) /
                     (trans[16 * i + 11] - trans[16 * i + 8] * tmp_real_w - trans[16 * i + 9] * tmp_real_h) ;
        double c1y = (trans[16 * i + 7]  - trans[16 * i + 4] * tmp_real_w - trans[16 * i + 5] * tmp_real_h) /
                     (trans[16 * i + 11] - trans[16 * i + 8] * tmp_real_w - trans[16 * i + 9] * tmp_real_h) ;
        double c2x = (trans[16 * i + 3]  + trans[16 * i]     * tmp_real_w - trans[16 * i + 1] * tmp_real_h) /
                     (trans[16 * i + 11] + trans[16 * i + 8] * tmp_real_w - trans[16 * i + 9] * tmp_real_h) ;
        double c2y = (trans[16 * i + 7]  + trans[16 * i + 4] * tmp_real_w - trans[16 * i + 5] * tmp_real_h) /
                     (trans[16 * i + 11] + trans[16 * i + 8] * tmp_real_w - trans[16 * i + 9] * tmp_real_h) ;
        double c3x = (trans[16 * i + 3]  + trans[16 * i]     * tmp_real_w + trans[16 * i + 1] * tmp_real_h) /
                     (trans[16 * i + 11] + trans[16 * i + 8] * tmp_real_w + trans[16 * i + 9] * tmp_real_h) ;
        double c3y = (trans[16 * i + 7]  + trans[16 * i + 4] * tmp_real_w + trans[16 * i + 5] * tmp_real_h) /
                     (trans[16 * i + 11] + trans[16 * i + 8] * tmp_real_w + trans[16 * i + 9] * tmp_real_h) ;
        double c4x = (trans[16 * i + 3]  - trans[16 * i]     * tmp_real_w + trans[16 * i + 1] * tmp_real_h) /
                     (trans[16 * i + 11] - trans[16 * i + 8] * tmp_real_w + trans[16 * i + 9] * tmp_real_h) ;
        double c4y = (trans[16 * i + 7]  - trans[16 * i + 4] * tmp_real_w + trans[16 * i + 5] * tmp_real_h) /
                     (trans[16 * i + 11] - trans[16 * i + 8] * tmp_real_w + trans[16 * i + 9] * tmp_real_h) ;

	    // reject transformations make marker too small in screen
        double two_area =   (c1x - c2x) * (c1y + c2y)
                         + (c2x - c3x) * (c2y + c3y)
                         + (c3x - c4x) * (c3y + c4y)
                         + (c4x - c1x) * (c4y + c1y);
        double area = std::abs(two_area / 2);

        float minx = std::min(c1x, std::min(c2x, std::min(c3x, c4x)));
        float maxx = std::max(c1x, std::max(c2x, std::max(c3x, c4x)));
        float miny = std::min(c1y, std::min(c2y, std::min(c3y, c4y)));
        float maxy = std::max(c1y, std::max(c2y, std::max(c3y, c4y)));

        const int margin = 1;
        if (area > area_thres && (minx >= margin) && (maxx <= iw-1-margin) && (miny >= margin) && (maxy <= ih-1-margin))
            insiders[i] = 1;
        else
            insiders[i] = 0;
    }
}

