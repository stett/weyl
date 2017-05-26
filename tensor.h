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
        template <size_t I, template<size_t...> class T, size_t... N>
        struct Reduction
        {
            template <size_t NM, size_t... M>
            using reduced_t = typename Reduction<I-1, T, N..., NM>::template reduced_t<M...>;
        };

        template <template<size_t...> class T, size_t... N>
        struct Reduction<0, T, N...>
        {
            template <size_t NM, size_t... M>
            using reduced_t = T<N..., M...>;
        };

        /// \struct ReducibleTensor
        /// \brief Provides a wrapper around the tensor template which may be used as a parameter for Reduction
        template <size_t... N>
        struct ReducibleTensor
        {
            template <typename T>
            using tensor_t = tensor<T, N...>;
        };

        template <>
        struct ReducibleTensor<>
        {
            template <typename T>
            using tensor_t = T;
        };

        /// \struct DoubleReducibleTensor
        /// \brief Provides a second rank reduction, for when two indexes must be removed from a tensor type.
        template <size_t J>
        struct DoubleReducibleTensor
        {
            template <size_t... N>
            struct Sub
            {
                // Note that in "J-1", the offset by one corrects for the fact that we are storing
                // the index of the second reduction in the parameter pack which is reduced by the
                // first one. Thus it will be one off.
                template <typename T>
                using tensor_t = typename Reduction< J - 1, ReducibleTensor >::template reduced_t<N...>::template tensor_t<T>;
            };
        };

        /// \struct TensorConvolution
        /// \brief Uses double rank reduction to determine the resultant type of a tensor convolution
        template <size_t Rank, size_t I, size_t J, typename T, size_t... N>
        struct TensorConvolution
        {
            using tensor_t = typename Reduction< I, DoubleReducibleTensor< Rank + J >::template Sub >::template reduced_t<N...>::template tensor_t<T>;
        };

        /// \struct Sum
        /// \brief Use template expansion to perform tensor convolution.
        template <size_t OuterI, size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                for (size_t i = 0; i < detail::Dimension<OuterI, N...>::value; ++i)
                    Sum< OuterI+1, OuterJ, I, J, RankN, RankM >::partial(result[i], part_a[i], part_b, a, b, inner);
            }
        };

        template <size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< I, OuterJ, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                Sum< I+1, OuterJ, I, J, RankN, RankM >::partial(result, part_a[inner], part_b, a, b, inner);
            }
        };

        template <size_t OuterJ, size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, OuterJ, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                for (size_t j = 0; j < detail::Dimension<OuterJ, M...>::value; ++j)
                    Sum< RankN, OuterJ+1, I, J, RankN, RankM >::partial(result[j], part_a, part_b[j], a, b, inner);
            }
        };

        template <size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, J, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                Sum< RankN, J+1, I, J, RankN, RankM >::partial(result, part_a, part_b[inner], a, b, inner);
            }
        };

        template <size_t I, size_t J, size_t RankN, size_t RankM>
        struct Sum< RankN, RankM, I, J, RankN, RankM >
        {
            template <typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
            static void partial(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
                result += part_a * part_b;
            }
        };
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

        tensor<T, N0, N...>& operator*=(const T& value) {
            for (size_t i = 0; i < N0; ++i) data[i] *= value; return *this;
        }

        tensor<T, N0, N...> operator*(const T& value) const {
            tensor<T, N0, N...> result(*this); result *= value; return result;
        }

        tensor<T, N0, N...>& operator+=(const tensor<T, N0, N...>& other) {
            for (size_t i = 0; i < N0; ++i) data[i] += other.data[i]; return *this;
        }

        tensor<T, N0, N...> operator+(const tensor<T, N0, N...>& other) const {
            tensor<T, N0, N...> result(*this); result += other; return result;
        }

        tensor<T, N0, N...>& operator-=(const tensor<T, N0, N...>& other) {
            for (size_t i = 0; i < N0; ++i) data[i] -= other.data[i]; return *this;
        }

        tensor<T, N0, N...> operator-(const tensor<T, N0, N...>& other) const {
            tensor<T, N0, N...> result(*this); result -= other; return result;
        }

    private:
        tensor<T, N...> data[N0];
    };


    template <typename T, size_t N>
    class tensor<T, N>
    {
    public:

        template <size_t I> // In this degenerative case, 'I' should always be zero.
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

        tensor<T, N>& operator*=(const T& value) {
            for (size_t i = 0; i < N; ++i) data[i] *= value; return *this;
        }

        tensor<T, N> operator*(const T& value) const {
            tensor<T, N> result(*this); result *= value; return result;
        }

        tensor<T, N>& operator+=(const tensor<T, N>& other) {
            for (size_t i = 0; i < N; ++i) data[i] += other.data[i]; return *this;
        }

        tensor<T, N> operator+(const tensor<T, N>& other) const {
            tensor<T, N> result(*this); result += other; return result;
        }

        tensor<T, N>& operator-=(const tensor<T, N>& other) {
            for (size_t i = 0; i < N; ++i) data[i] -= other.data[i]; return *this;
        }

        tensor<T, N> operator-(const tensor<T, N>& other) const {
            tensor<T, N> result(*this); result -= other; return result;
        }

    private:
        T data[N];
    };

    /// \brief Produce the tensor which is the convolution of two tensors.
    template <size_t I, size_t J, typename T, size_t... N, size_t... M>
    typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t
    sum(const tensor<T, N...>& a, const tensor<T, M...>& b) {

        // Determine the resulting tensor type, create an instance of it,
        // and initialize all of its values to "zero".
        using tensor_t = typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t;
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

    template <typename T, size_t... N>
    tensor<T, N...> operator*(const T& value, const tensor<T, N...>& tens) {
        return tens * value;
    }
}