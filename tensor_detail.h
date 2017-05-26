#pragma once

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
}