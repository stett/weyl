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
        template <size_t I, size_t J, size_t... M>
        typename detail::TensorConvolution<rank::value, I, J, T, N0, N..., M...>::tensor_t
        sum(const tensor<T, M...>& other) {
            static_assert(
                (int)dimension<I>::value == (int)detail::Dimension<J, M...>::value,
                "Indexes over which to sum must have equal dimensionality between tensors.");

            // Generate the convoluted tensor type and start with an empty one
            using tensor_t = typename detail::TensorConvolution<rank::value, I, J, T, N0, N..., M...>::tensor_t;
            return tensor_t();
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

    private:
        tensor<T, N...> data[N0];
    };


    template <typename T, size_t N>
    class tensor<T, N>
    {
    public:

        template <size_t I> // In this degenerative case, I should always be zero.
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

        template <size_t I, size_t J>
        T sum(const tensor<T, N>& other) {
            static_assert(I == 0 && J == 0, "Both rank indexes must be zero in the degenerative template.");
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

    /*
    template <size_t I, size_t J, typename DataResult, typename T, size_t... N, size_t... M>
    void partial_sum(DataResult& result, const tensor<T, N...>& a, const tensor<T, M...>& b) {
        partial_sum<I, J, 0, 0>(result, a, b, a, b);
    }

    template <size_t I, size_t J, size_t IndexN, size_t IndexM, typename DataResult, typename DataA, typename DataB, typename T, size_t... N, size_t... M>
    void partial_sum(DataResult& result, DataA& data_a, DataB& data_b, const tensor<T, N...>& a, const tensor<T, M...>& b) {

        constexpr size_t rank_n = detail::Rank<N...>::value;
        constexpr size_t rank_m = detail::Rank<N...>::value;
        constexpr size_t dimension_n = detail::Dimension<IndexN % rank_n, N...>::value;
        constexpr size_t dimension_m = detail::Dimension<IndexM % rank_m, M...>::value;

        if (IndexN < rank_n-2) {
            if (IndexN != I)
                for (size_t i = 0; i < dimension_n; ++i)
                    partial_sum<I, J, IndexN+1, IndexM>(result[i], data_a[i], data_b, a, b);
            else partial_sum<I, J, IndexN+1, IndexM>(result, data_a, data_b, a, b);

        } else if (IndexM < rank_m-2) {
            if (IndexM != J)
                for (size_t j = 0; j < dimension_m; ++j)
                    partial_sum<I, J, IndexN, IndexM+1>(result[j], data_a, data_b[j], a, b);
            else partial_sum<I, J, IndexN, IndexM+1>(result, data_a, data_b, a, b);

        } else {
        }
    }

    template <size_t I, size_t J, typename T, size_t... N, size_t... M>
    typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t
    sum(const tensor<T, N...>& a, const tensor<T, M...>& b) {
        using tensor_t = typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t;
        tensor_t result;
        partial_sum<I, J>(result, a, b);
        return result;
    }
    */

    /*
    template <size_t Sum, size_t Dim>
    struct SuperSum
    {
        template <typename ResultT>
        static void compute(ResultT& result) {
            SuperSum<Sum+1, Dim>::compute<>(result);
        }
    };

    template <size_t Dim>
    struct SuperSum<Dim, Dim>
    {
        template <typename ResultT>
        static void compute(ResultT& result) {

        }
    };

    template <size_t I, size_t J, size_t ISum, size_t JSum>
    struct Sum
    {
        template <typename DataResult, typename DataA, typename DataB, size_t... N, size_t... M>
        void compute(DataResult& result, const DataA& a, const DataB& b, size_t ij) {
            for (size_t i = 0; i < detail::Dimension<I, N...>::value; ++i) {
                Sum<I+1, J, ISum, JSum>::compute<>(result[i], a[i], b, ij);
            }
        }
    };

    template <size_t J, size_t ISum, size_t JSum>
    struct Sum< detail::Rank<N...>::value - 1, J, ISum, JSum>
    {
        template <typename DataResult, typename T, size_t... N, size_t... M>
        void compute(DataResult& result, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t ij) {
            for (size_t i = 0; i < detail::Dimension<I, N...>::value; ++i) {
                
            }
        }
    };

    */

    template <size_t OuterI, size_t OuterJ, size_t I, size_t J, typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
    void partial_sum(ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
        for (size_t i = 0; i < detail::Dimension<OuterI, N...>::value; ++i)
            partial_sum< OuterI+1, OuterJ, I, J >(result[i], part_a[i], part_b, a, b, inner);
    }

    template <size_t OuterJ, size_t I, size_t J, typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
    void partial_sum< I, OuterJ, I, J, ResultT, PartA, PartB, T, N..., M... >
    (ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
        partial_sum< OuterI+1, OuterJ, I, J >(result, part_a[inner], part_b, a, b, inner);
    }

    template <size_t OuterJ, size_t I, size_t J, typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
    void partial_sum< detail::Rank<N...>::value, OuterJ, I, J, ResultT, PartA, PartB, T, N..., M... >
    (ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
        for (size_t j = 0; j < detail::Dimension<OuterJ, M...>::value; ++j)
            partial_sum< OuterI, OuterJ+1, I, J >(result[j], part_a, part_b[j], a, b, inner);
    }

    template <size_t I, size_t J, typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
    void partial_sum< detail::Rank<N...>::value, J, I, J, ResultT, PartA, PartB, T, N..., M... >
    (ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
        partial_sum< OuterI, OuterJ+1, I, J >(result, part_a, part_b[inner], a, b, inner);
    }

    template <size_t I, size_t J, typename ResultT, typename PartA, typename PartB, typename T, size_t... N, size_t... M>
    void partial_sum< detail::Rank<N...>::value, detail::Rank<M...>::value, I, J, ResultT, PartA, PartB, T, N..., M... >
    (ResultT& result, const PartA& part_a, const PartB& part_b, const tensor<T, N...>& a, const tensor<T, M...>& b, size_t inner) {
        result += part_a * part_b;
    }

    template <size_t I, size_t J, typename T, size_t... N, size_t... M>
    typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t
    sum(const tensor<T, N...>& a, const tensor<T, M...>& b) {
        using tensor_t = typename detail::TensorConvolution< detail::Rank<N...>::value, I, J, T, N..., M... >::tensor_t;
        tensor_t result;

        for (size_t inner = 0; inner < detail::Dimension<I, N...>::value; ++inner)
            partial_sum<0, 0, I, J>(result, a, b, a, b, inner);

        return result;
    }
}