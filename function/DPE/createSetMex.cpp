#include <mex.h>
#include <cmath>

// Create the epsilon cover set
//
// Usage:
//   poses = createSetMex(min_rz, max_rz, min_rx, max_rx, min_tz, max_tz, ...
//                        step_rz0, step_rx, step_rz1, step_tx, step_ty, step_tz, ...
//                        fx, fy, cx, cy, tmp_real_w, tmp_real_h);                       
//
// Inputs:
//   min_rz     = lower bound of rz
//   max_rz     = upper bound of rz
//   min_rx     = lower bound of rx
//   max_rx     = upper bound of rx
//   min_tz     = lower bound of tz
//   max_tz     = upper bound of tz
//   step_rz0   = step rz0
//   step_rx    = step rx
//   step_rz1   = step rz1
//   step_tx    = step tx
//   step_ty    = step ty
//   step_tz    = step tz
//   fx         = focal length along the X-axis
//   fy         = focal length along the Y-axis
//   cx         = x coordinate of the camera's principle point 
//   cy         = y coordinate of the camera's principle point 
//   tmp_real_w = template width in real unit
//   tmp_real_h = template height in real unit
//
// Outputs:
//   poses     = all candidate poses (6*n)

void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[]) {
    // Retrieve the input data
    double min_rz = mxGetScalar(prhs[0]);
    double max_rz = mxGetScalar(prhs[1]);
    double min_rx = mxGetScalar(prhs[2]);
    double max_rx = mxGetScalar(prhs[3]);
    double min_tz = mxGetScalar(prhs[4]);
    double max_tz = mxGetScalar(prhs[5]);
    double step_rz0 = mxGetScalar(prhs[6]);
    double step_rx  = mxGetScalar(prhs[7]);
    double step_rz1 = mxGetScalar(prhs[8]);
    double step_tx  = mxGetScalar(prhs[9]);
    double step_ty  = mxGetScalar(prhs[10]);
    double step_tz  = mxGetScalar(prhs[11]);
    double fx = mxGetScalar(prhs[12]);
    double fy = mxGetScalar(prhs[13]);
    double cx = mxGetScalar(prhs[14]);
    double cy = mxGetScalar(prhs[15]);
    double tmp_real_w = mxGetScalar(prhs[16]);
    double tmp_real_h = mxGetScalar(prhs[17]);

    double mid_tz = sqrt(max_tz*min_tz);

    // Pre-calculate output matrix size
    double tz = min_tz;
    int num_poses = 0;
    double length = sqrt(tmp_real_w*tmp_real_w + tmp_real_h*tmp_real_h);
    while (tz <= max_tz) {
        double rz1 = min_rz;
        while (rz1 <= max_rz) {
            double rx = min_rx;
            while (rx <= max_rx) {
                double rz0 = min_rz;
                while (rz0 <= max_rz) {
                    double bound_tx = cx * tz / fx - tmp_real_h;
                    double tx = -bound_tx;
                    while (tx <= bound_tx) {
                        double bound_ty = cy * tz / fy - tmp_real_h;
                        double ty = -bound_ty;
                        while (ty < bound_ty) {
                            num_poses++;
                            ty += step_ty * (tz - length*sin(rx));
                        }
                        tx += step_tx * (tz - length*sin(rx));
                    }
                    rz0 += step_rz0;
                    if (rx == 0)
                        rz0 = max_rz + 1;
                }
                // we set tz to be 2 in asin_value for the sake of accuracy
                double asin_value = 2 - 1 / (1 / (2 - sin(rx)) + step_rx);
                if (asin_value <= 1 && asin_value >= -1)
                    rx = asin(asin_value);
                else
                    rx = max_rx + 1;
            }
            rz1 += step_rz1;
        }
        tz += tz*tz * step_tz / (1 - step_tz*tz);
    }

    plhs[0] = mxCreateDoubleMatrix(6, num_poses, mxREAL);  // 6 for (tx, ty, tz, rz0, rx, rz1)
    double *poses = mxGetPr(plhs[0]);

    // main loop
    int index = 0;
    tz = min_tz;
    while (tz <= max_tz) {
        double rz1 = min_rz;
        while (rz1 <= max_rz) {
            double rx = min_rx;
            while (rx <= max_rx) {
                double rz0 = min_rz;
                while (rz0 <= max_rz) {
                    double bound_tx = cx * tz / fx - tmp_real_h;
                    double tx = -bound_tx;
                    while (tx <= bound_tx) {
                        double bound_ty = cy * tz / fy - tmp_real_h;
                        double ty = -bound_ty;
                        while (ty < bound_ty) {
                            // add pi to rx for showing front side of the target plane
                            poses[index++] = rz0;
                            poses[index++] = rx;
                            poses[index++] = rz1;
                            poses[index++] = tx;
                            poses[index++] = ty;
                            poses[index++] = tz;
                            ty += step_ty * (tz - length*sin(rx));
                        }
                        tx += step_tx * (tz - length*sin(rx));
                    }
                    rz0 += step_rz0;
                    if (rx == 0)
                        rz0 = max_rz + 1;
                }
                double asin_value = 2 - 1 / (1 / (2 - sin(rx)) + step_rx);
                if (asin_value <= 1 && asin_value >= -1)
                    rx = asin(asin_value);
                else
                    rx = max_rx + 1;
            }
            rz1 += step_rz1;
        }
        tz += tz*tz * step_tz / (1 - step_tz*tz);
    }
}
