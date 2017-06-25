# Weyl

Weyl is a single-header library for doing math with tensors of any finite rank and dimension. It's named after [this guy](https://en.wikipedia.org/wiki/Weyl).

## Usage

### Examples: Tensor Convolutions

#### Example 1

The sum across the only index of two single-index tensors, if the indexes are 3-dimensional, is the same as the dot product of two 3-dimensional vectors.

    [ a b c ] * [ d e f ] = (a * d) + (b * e) + (c * f)

or

    tensor<float, 3> abc({ a, b, c});
    tensor<float, 3> def({ d, e, f});
    float product = weyl::sum<0, 0>(abc, def);

The `sum` function template will expand to the following.

    value = 0.0f;
    value += abc[0] * def[0];
    value += abc[1] * def[1];
    value += abc[2] * def[2];
    return value;

#### Example 2

The sum across the second index of a 3x3 tensor with the first index of another 3x3 tensor is essentially a matrix product.

    [ a b c ]   [ j k l ]   [ (a*j)+(b*m)+(c*p)  (a*k)+(b*n)+(c*q)  (a*l)+(b*o)+(c*r) ]
    [ d e f ] * [ m n o ] = [ (d*j)+(e*m)+(f*p)  (d*k)+(e*n)+(f*q)  (d*l)+(e*o)+(f*r) ]
    [ g h i ]   [ p q r ]   [ (g*j)+(h*m)+(i*p)  (g*k)+(h*n)+(i*q)  (g*l)+(h*o)+(i*r) ]

or

    tensor<float, 3, 3> a({ { ... }, { ... }, { ... } });
    tensor<float, 3, 3> b({ { ... }, { ... }, { ... } });
    tensor<float, 3, 3> product = weyl::sum<1, 0>(a, b);

In this case, `sum` will expand to something equivalent to the following, but with unrolled loops.

    tensor<float, 3, 3> value(0.0f);
    for (size_t n = 0; n < 3; ++n) // 1st dimension of a (2nd skipped - it is in the sum)
    for (size_t m = 0; m < 3; ++m) // 2nd dimension of b (1st skipped - it is in the sum)
    for (size_t i = 0; i < 3; ++i) // index for the 2nd and 1st dimensions of a and b respectively
        value[n][m] += a[n][i] * b[i][m];

## Testing

The library has been tested on Windows with MinGW using the following command from the repository root.

    g++ -std=c++14 tests/test.cpp -I. && .\a.exe

## Documentation

To compile html documentation, run `doxygen` in the repository root directory.

## Notes

Despite the ideal of high-ranking generalizations, for convenience Weyl includes a number of special functions for tensors of particular rank and dimensionality. For example `magnitude` will return the length of a first-rank tensor (vector). These functions remain because they are useful, but they may be removed in future versions of the library.
