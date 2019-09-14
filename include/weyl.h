#pragma once
#ifndef _WEYL_H_
#define _WEYL_H_

#ifndef WEYL_FLOAT_T
#define WEYL_FLOAT_T float
#endif

#ifndef WEYL_INT_T
#define WEYL_INT_T int
#endif

#include "weyl/tensor.h"
#include "weyl/vector.h"
#include "weyl/matrix.h"
#include "weyl/quaternion.h"

namespace weyl
{
    using float_t = WEYL_FLOAT_T;
    using int_t = WEYL_INT_T;

    template <size_t N>
    using vec = vector<float_t, N>;

    template <size_t N>
    using ivec = vector<int_t, N>;

    template <size_t N, size_t M>
    using mat = matrix<float_t, N, M>;

    using quat = quaternion<float_t>;

    using vec2 = vec<2>;
    using vec3 = vec<3>;
    using vec4 = vec<4>;

    using ivec2 = ivec<2>;
    using ivec3 = ivec<3>;
    using ivec4 = ivec<4>;

    using mat2 = mat<2, 2>;
    using mat3 = mat<3, 3>;
    using mat4 = mat<4, 4>;
}

#endif
