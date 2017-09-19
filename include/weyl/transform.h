#pragma once
#include "weyl/vector.h"
#include "weyl/matrix.h"
#include "weyl/quaternion.h"

namespace weyl
{
    template <typename T>
    struct transform
    {
        vector<T, 3> translation;
        vector<T, 3> scale;
        quaternion<T> rotation;

        matrix<T, 4, 4> matrix() {
            return matrix<T, 4, 4>({
                { scale[0], 0.0f, 0.0f, 0.0f },
                { 0.0f, scale[1], 0.0f, 0.0f },
                { 0.0f, 0.0f, scale[2], 0.0f },
                { 0.0f, 0.0f, 0.0f, 1.0f }
            }) * (
                (operator matrix<T, 4, 4>)rotation
            ) * matrix<T, 4, 4>({
                { 0.0f, 0.0f, 0.0f, translation[0] },
                { 0.0f, 0.0f, 0.0f, translation[1] },
                { 0.0f, 0.0f, 0.0f, translation[2] },
                { 0.0f, 0.0f, 0.0f, 1.0f }
            });
        }

        vector<T, 3> operator*(const vector<T, 3>& operand) const {
            return (rotation * (scale * operand)) + translation;
        }
    };
}
