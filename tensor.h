#pragma once
#include <initializer_list>
#include <algorithm>

namespace weyl
{
    template <typename T, size_t N0, size_t... N>
    class tensor;

    namespace detail
    {
        /// \struct Dimension
        /// \brief Given a set of dimensions N_0 through N_m, pick the Ith one and house its depth in value.
        ///
        /// In other words, Dimension<1, 8, 13, 22>::value == 8
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

        /// \struct Rank
        /// \brief Given a set of dimensions N_0 through N_m, house the rank, m.
        template <size_t N0, size_t... N>
        struct Rank
        {
            enum { value = 1 + (int)Rank<N...>::value };
        };

        template <size_t N>
        struct Rank<N>
        {
            enum { value = 1 };
        };

        /// \struct Reduction
        /// \brief Given a set of dimensions N through M, remove the Ith one, and house a tensor of the resulting dimensionality.
        template <size_t I, size_t J, typename T, size_t... N>
        struct Reduction
        {
            //static_assert(I < J, "The lower index must come first in a double index reduction");
            template <size_t NM, size_t... M>
            struct Post
            {
                using reduced_tensor_t = typename Reduction<I-1, J, T, N..., NM>::template Post<M...>::reduced_tensor_t;
            };
        };

        template <size_t J, typename T, size_t... N>
        struct Reduction<0, J, T, N...>
        {
            template <size_t NM, size_t... M>
            struct Post
            {
                using reduced_tensor_t = typename Reduction<0, J-1, T, N..., NM>::template Post<M...>::reduced_tensor_t;
            };
        };

        template <typename T, size_t... N>
        struct Reduction<0, 0, T, N...>
        {
            template <size_t NM, size_t... M>
            struct Post
            {
                using reduced_tensor_t = tensor<T, N..., M...>;
            };
        };

        /*
        /// \struct Sum
        /// \brief 
        template <size_t I, size_t J, typename T, size_t... N>
        struct Sum
        {
            template <size_t... M>
            struct Operand
            {
                constexpr size_t OffsetJ = Rank<N...>::count + J;
                using result_t = Reduction<>
            };
        };
        */
    }

    template <typename T, size_t N0, size_t... N>
    class tensor
    {
    public:

        /// \brief Get the "size" of the Ith dimension of the tensor.
        template <size_t I>
        using dimension = detail::Dimension<I, N0, N...>;

        using rank = detail::Rank<N0, N...>;

        using initializer = typename std::initializer_list< typename tensor<T, N...>::initializer >;

        tensor() {}

        tensor(const T& value) {
            std::fill(data, data + N0, tensor<T, N...>(value));
        }

        tensor(const initializer& values) {
            std::copy(values.begin(), values.end(), data);
        }

        tensor(const tensor<T, N0, N...>& other) {
            std::copy(other.data, other.data + N0, data);
        }

        /// \brief Produce a tensor product by summing up dimension I
        /// of this tensor with dimension J of the other.
        template <size_t I, size_t J, size_t M0, size_t... M>
        //typename detail::Sum<T, I, J, N0, N0, M0, M0>::ResultT sum(const tensor<T, M0, M...>& other) {
        tensor<T, 2, 2> sum(const tensor<T, M0, M...>& other) {
            static_assert(
                (int)dimension<I>::value == (int)detail::Dimension<J, M0, M...>::value,
                "Indexes over which to sum must have equal dimensionality between tensors.");

            //return detail::Sum<T, I, J, N0, N0, M0, M0>::ResultT();
            return tensor<T, 2, 2>();
        }


        tensor<T, N...>& operator[](size_t i) {
            return data[i];
        }

        const tensor<T, N...>& operator[](size_t i) const {
            return data[i];
        }

        template <typename OtherT>
        bool operator==(const OtherT& other) const {
            return false;
        }

        bool operator==(const tensor<T, N0, N...>& other) const {
            //
            // TODO: Expand this loop statically...
            //
            bool result = true;
            for (size_t i = 0; i < N0; ++i)
                if (data[i] != other.data[i])
                    return false;
            return true;
        }

        template <typename OtherT>
        bool operator!=(const OtherT& other) const {
            return true;
        }

        bool operator!=(const tensor<T, N0, N...>& other) const {
            return !operator==(other);
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

        using rank = detail::Rank<N>;

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

        tensor(const tensor<T, N>& other) {
            std::copy(other.data, other.data + N, data);
        }

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
            // int rank = Rank<M0, M...>::count;
            // tensor<T, M[0], M[1], ..., M[J], ..., M[rank]> result;
            //

            for (size_t n = 0; n < N; ++n) {

            }
        }
        */

        template <size_t I, size_t J>
        T sum(const tensor<T, N>& other) {
            //static_assert(I == 0);
            //static_assert(J == 0);
            return sum(other);
        }

        T sum(const tensor<T, N>& other) {
            T total = data[0] * other.data[0];
            for (size_t i = 1; i < N; ++i)
                total += data[i] * other.data[i];
            return total;
        }

        T& operator[](size_t i) {
            return data[i];
        }

        const T& operator[](size_t i) const {
            return data[i];
        }

        template <typename OtherT>
        bool operator==(const OtherT& other) const {
            return false;
        }

        bool operator==(const tensor<T, N>& other) const {
            //
            // TODO: Expand this loop statically...
            //
            for (size_t i = 0; i < N; ++i)
                if (data[i] != other.data[i])
                    return false;
            return true;
        }

        template <typename OtherT>
        bool operator!=(const OtherT& other) const {
            return true;
        }

        bool operator!=(const tensor<T, N>& other) const {
            return !operator==(other);
        }

    private:
        T data[N];
    };

    namespace detail
    {
        /*
        template <typename T, size_t I, size_t J, size_t N0, size_t N, size_t M0, size_t M>
        struct Sum
        {
            using ResultT = tensor<T, 2, 2>;
        };
        */
    }
}