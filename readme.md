# [Weyl](https://en.wikipedia.org/wiki/Weyl)

Weyl is a library for doing non-sparse math with tensors of any finite rank.


## Development Notes

### Examples: Inner Products

#### Example 1

The sum across the only index of two single-index tensors, if the indexes are 3-dimensional, is the same as the dot product of two 3-dimensional vectors.

    [ a b c ] * [ d e f ] = (a * d) + (b * e) + (c * f)

or

    tensor<float, 3> abc({ a, b, c});
    tensor<float, 3> def({ d, e, f});
    tensor<float>::sum<0, 0>(abc, def);

The `sum` function ought to expand to the following.

    float value = 0.0f;
    for (size_t i = 0; i < 3; ++i)
        value += abc[i] * def[i];
    return value;

#### Example 2

The sum across the second index of a 3x3 tensor with the first index of another 3x3 tensor is essentially a matrix product.

    [ a b c ]   [ j k l ]   [ (a*j)+(b*m)+(c*p)  (a*k)+(b*n)+(c*q)  (a*l)+(b*o)+(c*r) ]
    [ d e f ] * [ m n o ] = [ (d*j)+(e*m)+(f*p)  (d*k)+(e*n)+(f*q)  (d*l)+(e*o)+(f*r) ]
    [ g h i ]   [ p q r ]   [ (g*j)+(h*m)+(i*p)  (g*k)+(h*n)+(i*q)  (g*l)+(h*o)+(i*r) ]

or

    tensor<float, 3, 3> a({ { ... }, { ... }, { ... } });
    tensor<float, 3, 3> b({ { ... }, { ... }, { ... } });
    tensor<float>::sum<1, 0>(a, b);

In this case, `sum` should expand to something equivalent to the following.

    tensor<float, 3, 3> value(0.0f);
    for (size_t n = 0; n < 3; ++n) // 1st dimension of a (2nd skipped - it is in the sum)
    for (size_t m = 0; m < 3; ++m) // 2nd dimension of b (1st skipped - it is in the sum)
    for (size_t i = 0; i < 3; ++i) // index for the 2nd and 1st dimensions of a and b respectively
        value[n][m] += a[n][i] * b[i][m];

or, more generally

    size_t a_dim = a.dimension<0>::value;
    size_t b_dim = b.dimension<1>::value;
    size_t sum_dim = a.dimension<1>::value; // must be equivalent to b.dimension<0>::value (since we're summing over indexes 1 and 0 of a and b respectively)
    tensor<float, a_dim, b_dim> value(0.0f);
    for (size_t n = 0; n < a_dim; ++n) // 1st dimension of a (2nd skipped - it is in the sum)
    for (size_t m = 0; m < b_dim; ++m) // 2nd dimension of b (1st skipped - it is in the sum)
    for (size_t i = 0; i < sum_dim; ++i) // index for the 2nd and 1st dimensions of a and b respectively
        value[n][m] += a[n][i] * b[i][m];

From this point we can start to see patterns and hopefully think of ways to design the `sum` template so that it expands in the manner described above.

#### Example 3

By extension of the previous examples, we can "unroll" a higher dimensional tensor product. For example, the product of the second and first indexes respectively of the following tensors of dimensionality 2x2x2 and 3, is as follows.

        [ [ a b ]  [ e f ] ]
        [ [ c d ]  [ g h ] ]
    A = [                  ]
        [ [ i j ]  [ m n ] ]
        [ [ k l ]  [ o p ] ]

    B = [ q r ]

    C = sum<1, 0>(A, B)

      = [  ]

### Examples: Outer Products

The outer product is a coalescing of two tensors into a new, higher dimensional one.

#### Example 1

    A = [ a b ]

    B = [ c d ]

    A (*) B = [ a b ]
              [ c d ]
