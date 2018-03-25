#pragma once

#include <chrono>
#include <vector_functions.h>
#include <vector_types.h>
#include <thrust/random.h>
#include <thrust/tuple.h>

namespace pr {

const float EPS = 1e-25f;

struct R12t {
    float r11, r12, t1;
    float r21, r22, t2;
    float r31, r32, t3;
};

struct JacobRr {
    float j11, j12, j13;
    float j21, j22, j23;
    float j31, j32, j33;
    float j41, j42, j43;
    float j51, j52, j53;
    float j61, j62, j63;
};

struct Region {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
};

class Timer {
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> Clock;
public:
    Timer() { Reset(); }

    void Start() {
        running_ = true;
        prev_start_ = Now();
    }

    void Pause() {
        if (running_) {
            running_ = false;
            auto diff = Now() - prev_start_;
            count_ += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
        }
    }

    void Reset() {
        running_ = false;
        count_ = 0;
    }

    long long GetCount() {
        return count_;
    }

private:
    Clock Now() { return std::chrono::high_resolution_clock::now(); }

    long long count_;
    bool running_;
    Clock prev_start_;
};

struct RpParams {
    // about camera
    cv::Mat in_mat;

    // about template normalization matrix
    cv::Mat nm_mat;

    // about template image
    int tw, th;
    float tmp_real_w, tmp_real_h;

    // about camera frame
    int iw, ih;

    // number of pyramid levels
    int prm_lvls;
};

struct ValidTmpCoorFunctor {
    __host__ __device__
    bool operator()(const thrust::tuple<float4, float2, bool>& a) {
        return (thrust::get<2>(a) == 0);
    };
};

}  // namespace pr
