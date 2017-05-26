#include <cmath>
#include "tensor.h"

namespace weyl
{
    template <typename T, size_t N>
    class vector : public tensor<T, N>
    {
    public:
        template <typename... Args>
        vector(Args... args) : tensor<T, N>( static_cast<T>(args)... ) { }
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

    template <typename T, size_t N>
    T magnitude_sq(const vector<T, N>& v) {
        return dot(v, v);
    }

    template <typename T, size_t N>
    T magnitude(const vector<T, N>& v) {
        return std::sqrt(dot(v, v));
    }
}