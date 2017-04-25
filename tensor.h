#pragma once
#include <initializer_list>
#include <algorithm>

namespace weyl
{
    namespace detail
    {
        template <size_t I, size_t N0, size_t... N>
        struct Dimension
        {
            enum { value = Dimension<I-1, N...>::value };
        };

        template <size_t N0, size_t... N>
        struct Dimension<0, N0, N...>
        {
            enum { value = N0 };
        };

        template <size_t N0, size_t... N>
        struct Dimensions
        {
            enum { count = 1 + (int)Dimensions<N...>::count };
        };

        template <size_t N>
        struct Dimensions<N>
        {
            enum { count = 1 };
        };
    }

    template <typename T, size_t N0, size_t... N>
    class tensor
    {
    public:

        /// \brief Get the "size" of the Ith dimension of the tensor.
        template <size_t I>
        using dimension = detail::Dimension<I, N0, N...>;

        using dimensions = detail::Dimensions<N0, N...>;

        using initializer = typename std::initializer_list< typename tensor<T, N...>::initializer >;

        tensor() {}

        tensor(const T& value) {
            std::fill(data, data + N0, tensor<T, N...>(value));
        }

        tensor(const initializer& values) {
            std::copy(values.begin(), values.end(), data);
        }

        /// \brief Produce a tensor product by summing up dimension I
        /// of this tensor with dimension J of the other.
        template <size_t I, size_t J, size_t M0, size_t... M>
        auto sum(const tensor<T, M0, M...>& other) {
            static_assert(
                (int)dimension<I>::value == (int)detail::Dimension<J, M0, M...>::value,
                "Indexes over which to sum must have equal dimensionality between tensors.");
        }

        tensor<T, N...>& operator[](size_t i) {
            return data[i];
        }

        const tensor<T, N...>& operator[](size_t i) const {
            return data[i];
        }

    private:
        tensor<T, N...> data[N0];
    };


    template <typename T, size_t N>
    class tensor<T, N>
    {
    public:

        /// \brief Get the "size" of the Ith dimension of the tensor.
        /// In this degenerate case, I should always be zero.
        template <size_t I>
        using dimension = detail::Dimension<I, N>;

        using dimensions = detail::Dimensions<N>;

        using initializer = std::initializer_list<T>;

        tensor() {}

        tensor(const T& value) {
            std::fill(data, data + N, value);
        }

        tensor(const initializer& values) {
            std::copy(values.begin(), values.end(), data);
        }

        template <typename... Args>
        tensor(Args... args) : data{ args... } {}

        /*
        /// \brief Produce a tensor product by summing up dimension I
        /// of this tensor with dimension J of the other.
        template <size_t I, size_t J, size_t M0, size_t... M>
        auto sum(const tensor<T, M0, M...>& other) {
            static_assert(I == 0);
            static_assert(
                (int)dimension<I>::value == (int)detail::Dimension<J, M0, M...>::value,
                "Indexes over which to sum must have equal dimensionality between tensors.");

            //
            // int dim = Dimensions<M0, M...>::count;
            // tensor<T, M[0], M[1], ..., M[J], ..., M[dim]> result;
            //

            for (size_t n = 0; n < N; ++n) {

            }
        }
        */

        T& operator[](size_t i) {
            return data[i];
        }

        const T& operator[](size_t i) const {
            return data[i];
        }

    private:
        T data[N];
    };
}