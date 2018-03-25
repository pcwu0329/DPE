#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stack>
#include <iomanip>
#include "prUtil.h"
#include "prCommon.h"

namespace pr {

cv::Mat getHomoMatFromInExNm(const cv::Mat &in_mat,
                             const cv::Mat &ex_mat,
                             const cv::Mat &nm_mat) {
    cv::Mat ex_mat_124 = (cv::Mat_<float>(3, 3) << ex_mat.at<float>(0, 0),
                                                   ex_mat.at<float>(0, 1),
                                                   ex_mat.at<float>(0, 3),
                                                   ex_mat.at<float>(1, 0),
                                                   ex_mat.at<float>(1, 1),
                                                   ex_mat.at<float>(1, 3),
                                                   ex_mat.at<float>(2, 0),
                                                   ex_mat.at<float>(2, 1),
                                                   ex_mat.at<float>(2, 3));
    cv::Mat H = in_mat * ex_mat_124 * nm_mat;
    H /= H.at<float>(2, 2);
    return H.clone();
}

cv::Mat getHomoMatFromInEx(const cv::Mat &in_mat,
                           const cv::Mat &ex_mat) {
    cv::Mat ex_mat_124 = (cv::Mat_<float>(3, 3) << ex_mat.at<float>(0, 0),
                                                   ex_mat.at<float>(0, 1),
                                                   ex_mat.at<float>(0, 3),
                                                   ex_mat.at<float>(1, 0),
                                                   ex_mat.at<float>(1, 1),
                                                   ex_mat.at<float>(1, 3),
                                                   ex_mat.at<float>(2, 0),
                                                   ex_mat.at<float>(2, 1),
                                                   ex_mat.at<float>(2, 3));
    cv::Mat H = in_mat * ex_mat_124;
    H /= H.at<float>(2, 2);
    return H.clone();
}


cv::Mat getAxisAngleFromRotationMatirx(const cv::Mat &R)
{
    float a = acosf((trace(R).val[0] - 1) / 2);
    cv::Mat r;
    if (a < EPS)
    {
        r = (cv::Mat_<float>(3, 1) << R.at<float>(2, 1) - R.at<float>(1, 2),
                                      R.at<float>(0, 2) - R.at<float>(2, 0),
                                      R.at<float>(1, 0) - R.at<float>(0, 1));
        r *= 0.5;
    }
    else if (a >(M_PI - EPS))
    {
        cv::Mat S = 0.5 * (R - (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1));
        float b = sqrt(S.at<float>(0, 0) + 1);
        float c = sqrt(S.at<float>(1, 1) + 1);
        float d = sqrt(S.at<float>(2, 2) + 1);
        if (b > EPS)
        {
            c = S.at<float>(1, 0) / b;
            d = S.at<float>(2, 0) / b;
        }
        else if (c > EPS)
        {
            b = S.at<float>(0, 1) / c;
            d = S.at<float>(2, 1) / c;
        }
        else
        {
            b = S.at<float>(0, 2) / d;
            c = S.at<float>(1, 2) / d;
        }
        r = (cv::Mat_<float>(3, 1) << b, c, d);
    }
    else
    {
        r = (cv::Mat_<float>(3, 1) << R.at<float>(2, 1) - R.at<float>(1, 2),
                                      R.at<float>(0, 2) - R.at<float>(2, 0),
                                      R.at<float>(1, 0) - R.at<float>(0, 1));
        r *= (a / 2 / sinf(a));
    }
    return r.clone();
}

cv::Mat getRotationMatrixFromAxisAngle(const cv::Mat &r)
{
    float rx, ry, rz;
    rx = r.at<float>(0);
    ry = r.at<float>(1);
    rz = r.at<float>(2);
    cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat W = getCrossProductMatrix(r);
    cv::Mat W2 = W * W;
    const float a = norm(r);
    if (a < EPS)
        return I + W + 0.5 * W2;
    else
        return I + W * sin(a) / a + W2 * (1 - cos(a)) / (a * a);
}

cv::Mat getAaParametersFromExtrinsicMatirx(const cv::Mat &ex_mat)
{
    cv::Mat p(6, 1, CV_32F);
    getAxisAngleFromRotationMatirx(ex_mat(cv::Rect(0, 0, 3, 3))).copyTo(p(cv::Rect(0, 0, 1, 3)));
    ex_mat(cv::Rect(3, 0, 1, 3)).copyTo(p(cv::Rect(0, 3, 1, 3)));
    return p.clone();
}

cv::Mat getExtrinsicMatrixFromAaParameters(const cv::Mat &p)
{
    cv::Mat ex_mat = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    getRotationMatrixFromAxisAngle(p(cv::Rect(0, 0, 1, 3))).copyTo(ex_mat(cv::Rect(0, 0, 3, 3)));
    p(cv::Rect_<float>(0, 3, 1, 3)).copyTo(ex_mat(cv::Rect_<float>(3, 0, 1, 3)));
    return ex_mat.clone();
}

cv::Mat getCrossProductMatrix(const cv::Mat &r)
{
    float rx = r.at<float>(0);
    float ry = r.at<float>(1);
    float rz = r.at<float>(2);
    cv::Mat W = (cv::Mat_<float>(3, 3) << 0, -rz, ry, rz, 0, -rx, -ry, rx, 0);
    return W.clone();
}

R12t getR12t(const cv::Mat &ex_mat)
{
    R12t R12_t;
    R12_t.r11 = ex_mat.at<float>(0, 0);
    R12_t.r12 = ex_mat.at<float>(0, 1);
    R12_t.t1  = ex_mat.at<float>(0, 3);
    R12_t.r21 = ex_mat.at<float>(1, 0);
    R12_t.r22 = ex_mat.at<float>(1, 1);
    R12_t.t2  = ex_mat.at<float>(1, 3);
    R12_t.r31 = ex_mat.at<float>(2, 0);
    R12_t.r32 = ex_mat.at<float>(2, 1);
    R12_t.t3  = ex_mat.at<float>(2, 3);
    return R12_t;
}

JacobRr getJacobRr(const cv::Mat &r)
{
    JacobRr J_Rr;

    const float &rx = r.at<float>(0);
    const float &ry = r.at<float>(1);
    const float &rz = r.at<float>(2);

    float a = norm(r);
    if (a < EPS) {
        J_Rr.j11 = 0;
        J_Rr.j12 = -ry;
        J_Rr.j13 = -rz;
        J_Rr.j21 = ry / 2;
        J_Rr.j22 = rx / 2;
        J_Rr.j23 = -1;
        J_Rr.j31 = ry / 2;
        J_Rr.j32 = rx / 2;
        J_Rr.j33 = 1;
        J_Rr.j41 = -rx;
        J_Rr.j42 = 0;
        J_Rr.j43 = -rz;
        J_Rr.j51 = rz / 2;
        J_Rr.j52 = -1;
        J_Rr.j53 = rx / 2;
        J_Rr.j61 = 1;
        J_Rr.j62 = rz / 2;
        J_Rr.j63 = ry / 2;
    }
    else {
        cv::Mat W = getCrossProductMatrix(r);
        cv::Mat R = getRotationMatrixFromAxisAngle(r);
        cv::Mat Id_R = cv::Mat::eye(3, 3, CV_32F) - R;
        cv::Mat M = W * Id_R;
        cv::Mat R_a2 = R / (a*a);
        cv::Mat dR_dr0 = (r.at<float>(0)*W + getCrossProductMatrix(M.col(0)))*R_a2;
        cv::Mat dR_dr1 = (r.at<float>(1)*W + getCrossProductMatrix(M.col(1)))*R_a2;
        cv::Mat dR_dr2 = (r.at<float>(2)*W + getCrossProductMatrix(M.col(2)))*R_a2;
        J_Rr.j11 = dR_dr0.at<float>(0, 0);
        J_Rr.j12 = dR_dr1.at<float>(0, 0);
        J_Rr.j13 = dR_dr2.at<float>(0, 0);
        J_Rr.j21 = dR_dr0.at<float>(0, 1);
        J_Rr.j22 = dR_dr1.at<float>(0, 1);
        J_Rr.j23 = dR_dr2.at<float>(0, 1);
        J_Rr.j31 = dR_dr0.at<float>(1, 0);
        J_Rr.j32 = dR_dr1.at<float>(1, 0);
        J_Rr.j33 = dR_dr2.at<float>(1, 0);
        J_Rr.j41 = dR_dr0.at<float>(1, 1);
        J_Rr.j42 = dR_dr1.at<float>(1, 1);
        J_Rr.j43 = dR_dr2.at<float>(1, 1);
        J_Rr.j51 = dR_dr0.at<float>(2, 0);
        J_Rr.j52 = dR_dr1.at<float>(2, 0);
        J_Rr.j53 = dR_dr2.at<float>(2, 0);
        J_Rr.j61 = dR_dr0.at<float>(2, 1);
        J_Rr.j62 = dR_dr1.at<float>(2, 1);
        J_Rr.j63 = dR_dr2.at<float>(2, 1);
    }
    return J_Rr;
}

float polygonArea(const cv::Mat &corners) {
    float area = 0.f;
    int num = corners.cols;
    int j = num - 1;
    for (int i = 0; i < num; ++i) {
        area += (corners.at<float>(0, j) + corners.at<float>(0, i))
               *(corners.at<float>(1, j) - corners.at<float>(1, i));
        j = i;
    }
    return std::abs(area) * 0.5f;
}

void calRegion(const cv::Mat &boundary,
               const cv::Mat &H,
               int tw,
               int th,
               int iw,
               int ih,
               float scale,
               cv::Mat *corners,
               Region *region) {
    (*corners) = H * boundary;
    auto w = (cv::Mat(1.f / (*corners)(cv::Rect(0, 2, 4, 1)))).clone();
    cv::multiply((*corners)(cv::Rect(0, 0, 4, 1)), w, (*corners)(cv::Rect(0, 0, 4, 1)));
    cv::multiply((*corners)(cv::Rect(0, 1, 4, 1)), w, (*corners)(cv::Rect(0, 1, 4, 1)));

    double x_min, x_max, y_min, y_max, x_center, y_center, x_length, y_length;
    cv::minMaxIdx((*corners)(cv::Rect(0, 0, 4, 1)), &x_min, &x_max);
    cv::minMaxIdx((*corners)(cv::Rect(0, 1, 4, 1)), &y_min, &y_max);
    x_center = (x_max + x_min) / 2.0;
    y_center = (y_max + y_min) / 2.0;
    x_length = x_max - x_center;
    y_length = y_max - y_center;
    x_max = x_center + x_length * scale;
    y_max = y_center + y_length * scale;
    x_min = x_center - x_length * scale;
    y_min = y_center - y_length * scale;
    region->x_max = int(std::min(ceil(x_max), iw - 1.0));
    region->y_max = int(std::min(ceil(y_max), ih - 1.0));
    region->x_min = int(std::max(floor(x_min), 0.0));
    region->y_min = int(std::max(floor(y_min), 0.0));
}

void calPoseDiff(const cv::Mat &p1,
                 const cv::Mat &p2,
                 float *diff_r,
                 float *diff_t)
{
    auto R1 = getRotationMatrixFromAxisAngle(p1(cv::Rect(0, 0, 1, 3))).clone();
    auto t1 = p1(cv::Rect(0, 3, 1, 3)).clone();
    auto R2 = getRotationMatrixFromAxisAngle(p2(cv::Rect(0, 0, 1, 3))).clone();
    auto t2 = p2(cv::Rect(0, 3, 1, 3)).clone();
    cv::Mat R2_transpose;
    transpose(R2, R2_transpose);
    *diff_r = acosf((trace(R2_transpose*R1).val[0] - 1) / 2) * 180.f / M_PI;
    *diff_t = norm(t1 - t2) / norm(t1) * 100;
}

void calDeltaP(float *JtJ_ptr,
               float *JtE_ptr,
               cv::Mat *delta_p,
               cv::Mat *JtE) {
    Eigen::Matrix<float, 6, 6> H = Eigen::Map<Eigen::Matrix<float, 6, 6>>(JtJ_ptr);
    Eigen::Matrix<float, 6, 1> delta_p_eigen = H.inverse()*Eigen::Map<Eigen::Matrix<float, 6, 1>>(JtE_ptr);
    for (int i = 0; i < 6; ++i) {
        (*delta_p).at<float>(i) = delta_p_eigen[i];
        (*JtE).at<float>(i) = JtE_ptr[i];
    }
}

}  // namespace pr
