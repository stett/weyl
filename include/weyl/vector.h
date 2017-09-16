#pragma once
#include "weyl/tensor.h"

namespace weyl
{
    template <typename T, size_t N>
    class vector : public tensor<T, N>
    {
    public:
        template <typename... Args>
        vector(Args... args) : tensor_t( static_cast<T>(args)... ) { }

        vector() : tensor_t() {}

        vector(const T& value) : tensor_t(value) { }

        vector(const initializer_t& values) : tensor_t(values) { }

        vector(const tensor_t& other) : tensor_t(other) { }

        vector(const vector<T, N - 1>& other, T value) {
            for (size_t i = 0; i < N - 1; ++i)
                _data[i] = other[i];
            _data[N - 1] = value;
        }

        vector(const vector<T, N + 1>& other) {
            for (size_t i = 0; i < N; ++i)
                _data[i] = other[i];
        }

        T magnitude_sq() const {
            return sum<0, 0>(*this, *this);
        }

        T magnitude() const {
            return sqrt(magnitude_sq());
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
