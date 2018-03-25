#include <mex.h>
#include <cmath>
#include <omp.h>

// Evaluate appearance error (intensity invariant mode)
//
// Usage:
//   distances = evaluateEaInvarMex(tmp_y, tmp_cb, tmp_cr, ...
//                                  img_y, img_cb, img_cr, ...
//                                  trans, xs, ys, tmp_real_w, tmp_real_h);                 
//
// Inputs:
//   tmp_y      = y channel of the template image
//   tmp_cb     = cb channel of the template image
//   tmp_cr     = cr channel of the template image
//   img_y      = y channel of the camera image
//   img_cb     = cb channel of the camera image
//   img_cr     = cr channel of the camera image
//   trans      = transformation matrices (16*n)
//   xs         = x coordinates in the template image (1*m)
//   ys         = y coordinates in the template image (1*m)
//   tmp_real_w = template width in real unit
//   tmp_real_h = template height in real unit
//
// Outputs:
//   distances = appearances errors (1*n)

void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[]) {
    // find the dimensions of the data
    int th = mxGetM(prhs[0]);
    int tw = mxGetN(prhs[0]);
    int ih = mxGetM(prhs[3]);
    int iw = mxGetN(prhs[3]);
    int num_poses = mxGetN(prhs[6]);
    int num_points = mxGetN(prhs[7]);

    // create an mxArray for the output data
    plhs[0] = mxCreateDoubleMatrix(1, num_poses, mxREAL);

    // create an mxArrays for temporary data
    double *tmp_real_xs = (double *)malloc(num_points * sizeof(double));
    double *tmp_real_ys = (double *)malloc(num_points * sizeof(double));
    double *tmp_y_vals  = (double *)malloc(num_points * sizeof(double));
    double *tmp_cb_vals = (double *)malloc(num_points * sizeof(double));
    double *tmp_cr_vals = (double *)malloc(num_points * sizeof(double));

    // retrieve the input data
    double *tmp_y = mxGetPr(prhs[0]);
    double *tmp_cb = mxGetPr(prhs[1]);
    double *tmp_cr = mxGetPr(prhs[2]);
    double *img_y = mxGetPr(prhs[3]);
    double *img_cb = mxGetPr(prhs[4]);
    double *img_cr = mxGetPr(prhs[5]);
    double *trans = mxGetPr(prhs[6]);
    int *xs = (int*)mxGetPr(prhs[7]);
    int *ys = (int*)mxGetPr(prhs[8]);
    double tmp_real_w = mxGetScalar(prhs[9]);
    double tmp_real_h = mxGetScalar(prhs[10]);
    
    // normalize template coordinates (transform to the real unit)
    double center_x = 0.5*(tw - 1);
    double center_y = 0.5*(th - 1);
    for (int i = 0; i < num_points; i++) {
        tmp_real_xs[i] = double(xs[i] - center_x) / (tw*0.5) * tmp_real_w;
        tmp_real_ys[i] = -double(ys[i] - center_y) / (th*0.5) * tmp_real_h;
    }

    // pre-calculating selected template pixel values
    for (int j = 0; j < num_points; j++) {
        tmp_y_vals [j] = tmp_y [(xs[j])*th + ys[j]];
        tmp_cb_vals[j] = tmp_cb[(xs[j])*th + ys[j]];
        tmp_cr_vals[j] = tmp_cr[(xs[j])*th + ys[j]];
    }

    // create a pointer to the output data
    double *distances = mxGetPr(plhs[0]);

    // main loop
    int i;
#pragma omp parallel for private(i) num_threads(8)
    for (i = 0; i < num_poses; i++) {
        double m11, m12, m14, m21, m22, m24, m31, m32, m34;
        m11 = trans[16 * i];
        m12 = trans[16 * i + 1];
        m14 = trans[16 * i + 3];
        m21 = trans[16 * i + 4];
        m22 = trans[16 * i + 5];
        m24 = trans[16 * i + 7];
        m31 = trans[16 * i + 8];
        m32 = trans[16 * i + 9];
        m34 = trans[16 * i + 11];

        double score = 0;
        double sum_of_tmp_val = 0;
        double sum_of_img_val = 0;
        double sum_of_sq_tmp_val = 0;
        double sum_of_sq_img_val = 0;       
        double* img_y_vals = (double *)malloc(num_points * sizeof(double));
        double* img_cb_vals = (double *)malloc(num_points * sizeof(double));
        double* img_cr_vals = (double *)malloc(num_points * sizeof(double));
        for (int j = 0; j < num_points; j++) {
            double denominator = m31*tmp_real_xs[j] + m32*tmp_real_ys[j] + m34;
            int coor_x = int((m11*tmp_real_xs[j] + m12*tmp_real_ys[j] + m14) / denominator + 0.5); // includes rounding
            int coor_y = int((m21*tmp_real_xs[j] + m22*tmp_real_ys[j] + m24) / denominator + 0.5); // includes rounding
            int img_index = coor_x*ih + coor_y;
            img_y_vals[j] = img_y[img_index];
            img_cb_vals[j] = img_cb[img_index];
            img_cr_vals[j] = img_cr[img_index];
            sum_of_tmp_val += tmp_y_vals[j];
            sum_of_img_val += img_y_vals[j];
            sum_of_sq_tmp_val += (tmp_y_vals[j]*tmp_y_vals[j]);
            sum_of_sq_img_val += (img_y_vals[j]*img_y_vals[j]);
        }

        double epsilon = 1e-7;
        double sig_tmp = sqrt((sum_of_sq_tmp_val - (sum_of_tmp_val*sum_of_tmp_val) / num_points) / num_points) + epsilon;
        double sig_img = sqrt((sum_of_sq_img_val - (sum_of_img_val*sum_of_img_val) / num_points) / num_points) + epsilon;
        double mean_tmp = sum_of_tmp_val / num_points;
        double mean_img = sum_of_img_val / num_points;
        double sig_tmp_over_sig_img = sig_tmp / sig_img;

        // variable that stores a sum used repeatadly in the computation: -mean_tmp+sig_tmp_over_sig_img*mean_img
        double faster = -mean_tmp + sig_tmp_over_sig_img * mean_img;
        for (int j = 0; j < num_points; j++) {
            score +=  0.50*fabs(tmp_y_vals[j] - sig_tmp_over_sig_img*img_y_vals[j] + faster)
                    + 0.25*fabs(tmp_cb_vals[j] - img_cb_vals[j])
                    + 0.25*fabs(tmp_cr_vals[j] - img_cr_vals[j]);
        }

        free(img_y_vals);
        free(img_cb_vals);
        free(img_cr_vals);

        distances[i] = score / num_points;
    }

    // free the allocated arrays
    free(tmp_real_xs);
    free(tmp_real_ys);
    free(tmp_y_vals);
    free(tmp_cb_vals);
    free(tmp_cr_vals);
}