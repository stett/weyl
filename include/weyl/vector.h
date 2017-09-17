#pragma once
#ifndef _WEYL_VECTOR_H_
#define _WEYL_VECTOR_H_

#include "weyl/tensor.h"

namespace weyl
{
    template <typename T, size_t N>
    class vector : public tensor<T, N>
    {
    public:
        vector() : tensor_t() {}

        template <typename... Values>
        vector(const T& value, Values... values) : tensor_t({ value, static_cast<T>(values)... }) { }

        vector(const initializer_t& values) : tensor_t(values) { }

        vector(const tensor_t& other) : tensor_t(other) { }

        template <size_t M>
        vector(const vector<T, M>& other) {
            /*
            for (size_t i = 0; i < std::min(N, M); ++i)
                _data[i] = other[i];
                */
        }

        vector(const vector<T, N - 1>& other, const T& extra) {
            /*
            for (size_t i = 0; i < N-1; ++i)
                _data[i] = other[i];
            _data[N - 1] = extra;
            */
        }

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        vector<T, N>& normalize() {
            T mag_inv = static_cast<T>(1) / magnitude();
            for (size_t i = 0; i < N; ++i)
                _data[i] *= mag_inv;
            return *this;
        }

        vector<T, N> normal() const {
            vector<T, N> result(*this);
            result.normalize();
            return result;
        }

        template<size_t I>
        vector<T, N / 2> half() const {
            vector<T, N / 2> result;
            for (size_t i = 0; i < N/2; ++i)
                result[i] = _data[i + (I * N / 2)];
            return result;
        }

        void split(vector<T, N/2>& a, vector<T, N/2>& b) const {
            static_assert(N%2 == 0, "Vector must have even dimensionality.");
            for (size_t i = 0; i < N/2; ++i) {
                a[i] = _data[i];
                b[i] = _data[i + N/2];
            }
        }
    };

    template <typename T>
    class vector<T, 2> : public tensor<T, 2>
    {
    public:
        vector() : tensor_t() { }
        vector(const T& v) : tensor_t({ v }) { }
        vector(const T& v0, const T& v1) : tensor_t({ v0, v1 }) { }
        vector(const initializer_t& values) : tensor_t(values) { }
        vector(const tensor_t& other) : tensor_t(other) { }

        template <size_t M>
        vector(const vector<T, M>& other) {
            for (size_t i = 0; i < min(2, M); ++i)
                _data[i] = other[i];
        }

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }

        template<size_t I>
        vector<T, 1> half() const {
            return vector<T, 1>(_data[I]);
        }

        void split(vector<T, 1>& a, vector<T, 1>& b) const {
            a[0] = _data[0];
            b[0] = _data[1];
        }
    };

    template <typename T>
    class vector<T, 1> : public tensor<T, 1>
    {
    public:
        vector() : tensor_t() { }
        vector(const T& v) : tensor_t(v) { }
        vector(const initializer_t& values) : tensor_t(values) { }
        vector(const tensor_t& other) : tensor_t(other) { }

        template <size_t M>
        vector(const vector<T, M>& other) {
            for (size_t i = 0; i < min(1, M); ++i)
                _data[i] = other[i];
        }

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
        }
    };

    template <typename T, size_t N>
    T dot(const vector<T, N>& a, const vector<T, N>& b) {
        return sum<0, 0>(a, b);
    }

    template <typename T>
    T cross(const vector<T, 2>& a, const vector<T, 2>& b) {
        return (a[0] * b[1]) - (a[1] * b[0]);
    }

    template <typename T>
    vector<T, 3> cross(const vector<T, 3>& a, const vector<T, 3>& b) {
        return vector<T, 3>({
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])
        });
    }
}

#endif
