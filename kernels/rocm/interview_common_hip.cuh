#pragma once
// Portable SIMT portion of interview/common.cuh. Matrix/async instructions are
// implemented by the separate native AMD headers, never by no-op PTX macros.
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
static constexpr int kWarpSize = 32;
#if !defined(__HIP_PLATFORM_AMD__)
#error "AMD HIP build required"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "This qualified port targets gfx1201 wave32"
#endif
