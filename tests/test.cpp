// Catch
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
//#include "weyl.h"
#include "test_tensor.cpp"
#include "test_vector.cpp"
#include "test_matrix.cpp"

int main( int argc, char* argv[] )
{
    // Run tests
    int result = Catch::Session().run( argc, argv );

    // TODO: Generate result badge
    bool success = result == 0;

    // Return the number of failed tests
    return ( result < 0xff ? result : 0xff );
}
