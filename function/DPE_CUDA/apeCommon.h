#pragma once

#include <chrono>
#include <vector_functions.h>
#include <vector_types.h>
#include <thrust/random.h>
#include <thrust/tuple.h>

namespace ape {

struct Pose {
    float rz0, rx, rz1, tx, ty, tz;
};

__inline__ __host__ __device__
Pose makePose(float rz0, float rx, float rz1,
               float tx, float ty, float tz)
{
    Pose pose;
    pose.rz0 = rz0;
    pose.rx = rx;
    pose.rz1 = rz1;
    pose.tx = tx;
    pose.ty = ty;
    pose.tz = tz;
    return pose;
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

struct ApeParams {
    void ShrinkNet(float factor) {
        epsilon *= factor;
        step.rz0 *= factor;
        step.rx *= factor;
        step.rz1 *= factor;
        step.tx *= factor;
        step.ty *= factor;
        step.tz *= factor;
    }

    void UpdateNet(const ApeParams &_apeParams) {
        epsilon = _apeParams.epsilon;
        step = _apeParams.step;
    }

    // about pose net
    float epsilon;
    float min_tz, max_tz;
    float min_rx, max_rx;
    float min_rz, max_rz;
    Pose step;

    // about camera
    float fx, fy;
    float cx, cy;

    // about template image
    int tw, th;
    float tmp_real_w, tmp_real_h;

    // about camera frame
    int iw, ih;
};

// random coordinate ([0, w-1], [0, h-1]) generator
struct CoorRngFunctor {
    int w, h;

    __host__ __device__
    CoorRngFunctor(int _w, int _h) : w(_w), h(_h) {}

    __host__ __device__
    int2 operator()(const int n) const {
        thrust::default_random_engine rng;
        // range = [a, b] (only valid for non-negative integer)
        // -1 would be regarded as 0, -2 as -1, and so on
        thrust::uniform_int_distribution<int> x_range(0, w - 1);
        thrust::uniform_int_distribution<int> y_range(0, h - 1);
        rng.discard(n);
        return make_int2(x_range(rng), y_range(rng));
    };
};

struct ValidFunctor {
    __host__ __device__
    bool operator()(const thrust::tuple<Pose, bool>& a) {
        return (thrust::get<1>(a) == 0);
    };
};

}  // namespace ape
