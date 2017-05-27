#pragma once
#include <initializer_list>
#include <algorithm>
#include "tensor_detail.h"

namespace weyl
{
    template <typename T, size_t N0, size_t... N>
    class tensor
    {
    public:

        /// \brief Get the number of indexes of the tensor
        using rank = detail::Rank<N0, N...>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        template <size_t I>
        using dimension = detail::Dimension<I, N0, N...>;

        using initializer = typename std::initializer_list< typename tensor<T, N...>::initializer >;

        using tensor_t = tensor<T, N0, N...>;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N0, N...>::value, I, J, T, N0, N..., M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(data, data + N0, tensor<T, N...>(value));
        }

        tensor(const initializer& values) {
            std::copy(values.begin(), values.end(), data);
        }

        tensor(const tensor_t& other) {
            std::copy(other.data, other.data + N0, data);
        }

        tensor<T, N...>& operator[](size_t i) {
            return data[i];
        }

        const tensor<T, N...>& operator[](size_t i) const {
            return data[i];
        }

        bool operator==(const tensor_t& other) const {
            bool result = true;
            for (size_t i = 0; i < N0; ++i)
                if (data[i] != other.data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < N0; ++i) data[i] *= value; return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this); result *= value; return result;
        }

        tensor_t& operator*=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i) data[i] *= other.data[i]; return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this); result *= other; return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i) data[i] += other.data[i]; return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor_t result(*this); result += other; return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < N0; ++i) data[i] -= other.data[i]; return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this); result -= other; return result;
        }

    protected:
        tensor<T, N...> data[N0];
    };


    template <typename T, size_t N>
    class tensor<T, N>
    {
    public:

        /// \brief Get the number of indexes of the tensor. In this case ::value will always be 1.
        using rank = detail::Rank<N>;

        /// \brief Get the number of dimensions represented by the Ith index of the tensor.
        // In this degenerative case, 'I' should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;

        using initializer = std::initializer_list<T>;

        using tensor_t = tensor<T, N>;

        /// \brief Determine the tensor type which would result from convolution with another tensor.
        template <size_t I, size_t J, size_t... M>
        using convolution_t = typename detail::TensorConvolution< detail::Rank<N>::value, I, J, T, N, M... >::tensor_t;

        tensor() {}

        tensor(const T& value) {
            std::fill(data, data + N, value);
        }

        tensor(const initializer& values) {
            std::copy(values.begin(), values.end(), data);
        }

        template <typename... Args>
        tensor(Args... args) : data{ args... } {}

        tensor(const tensor_t& other) {
            std::copy(other.data, other.data + N, data);
        }

        T& operator[](size_t i) {
            return data[i];
        }

        const T& operator[](size_t i) const {
            return data[i];
        }

        bool operator==(const tensor_t& other) const {
            for (size_t i = 0; i < N; ++i)
                if (data[i] != other.data[i])
                    return false;
            return true;
        }

        bool operator!=(const tensor_t& other) const {
            return !operator==(other);
        }

        tensor_t& operator*=(const T& value) {
            for (size_t i = 0; i < N; ++i) data[i] *= value; return *this;
        }

        tensor_t operator*(const T& value) const {
            tensor_t result(*this); result *= value; return result;
        }

        tensor_t& operator*=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i) data[i] *= other.data[i]; return *this;
        }

        tensor_t operator*(const tensor_t& other) const {
            tensor_t result(*this); result *= other; return result;
        }

        tensor_t& operator+=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i) data[i] += other.data[i]; return *this;
        }

        tensor_t operator+(const tensor_t& other) const {
            tensor<T, N> result(*this); result += other; return result;
        }

        tensor_t& operator-=(const tensor_t& other) {
            for (size_t i = 0; i < N; ++i) data[i] -= other.data[i]; return *this;
        }

        tensor_t operator-(const tensor_t& other) const {
            tensor_t result(*this); result -= other; return result;
        }

    protected:
        T data[N];
    };

    /// \brief Produce the tensor which is the convolution of two tensors.
    template <size_t I, size_t J, typename T, size_t... N, size_t... M>
    typename tensor<T, N...>::template convolution_t<I, J, M...>
    sum(const tensor<T, N...>& a, const tensor<T, M...>& b) {

        // Determine the resulting tensor type, create an instance of it,
        // and initialize all of its values to "zero".
        using tensor_t = typename tensor<T, N...>::template convolution_t<I, J, M...>;
        tensor_t result(static_cast<T>(0));

        // Do the summation - this loop corresponds to the Riemann sum in an ordinary tensor product.
        for (size_t inner = 0; inner < detail::Dimension<I, N...>::value; ++inner)
            detail::Sum< 0, 0, I, J, detail::Rank<N...>::value, detail::Rank<M...>::value >::partial(result, a, b, a, b, inner);

        // Return the convolution
        return result;
    }

    /// \brief Degenerate convolution case - mirrors a simple vector dot product.
    template <size_t I, size_t J, typename T, size_t N, size_t M>
    T sum(const tensor<T, N>& a, const tensor<T, M>& b) {
        static_assert(I == 0 && J == 0, "Both tensors have rank 1, so the first dimensions of each must be used in the convolution.");
        T result = static_cast<T>(0);
        for (size_t inner = 0; inner < N; ++inner)
            detail::Sum< 0, 0, I, J, 1, 1 >::partial(result, a, b, a, b, inner);
        return result;
    }

    /// \brief Scalar tensor product.
    template <typename T, size_t... N>
    tensor<T, N...> operator*(const T& value, const tensor<T, N...>& tens) {
        return tens * value;
    }
}