#pragma once

#include <span>

namespace pds {

    constexpr std::uint32_t numberDimensions = 2u;

    struct Point {
        float x;
        float y;
    };

    struct Parameters {
        // Bounds of the rectangle are 0 to xMaximum / yMaximum
        float xMaximum = 50.0f;
        float yMaximum = 50.0f;
        // The minimum distance between two points. Must be positive 
        // and reducing this will increase the point density and calculation time
        float minimumDistance = 1.0f;
        // Initial seed to allow for reproducable tests and benchmarks
        uint32_t seed = 42;

        // Returns the upper bound for the cell count for the backing
        // uniform grid which can be used to preallocate the storage
        uint32_t getMaximumCellCount() const;

        // Returns the upper bound for the result count which can
        // be used this to preallocate the result storage
        uint32_t getMaximumResultCount() const;
    };

    // Implementation of the fast poisson disk sampling from https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    // All allocations are removed and the caller is responsible for allocating memory of the respective size
    // Use the upper bound methods from the parameter struct to allocate accordingly
    // The function returns the number of found points which are stored in the resultingPoints
    uint32_t fastPoissonDiskSampling(
        std::span<Point> resultingPoints,
        std::span<int32_t> workingBufferGrid,
        std::span<int32_t> workingBufferActiveSet,
        const Parameters& parameters);

    // Implementation of the maixmal poisson disk sampling from http://extremelearning.com.au/an-improved-version-of-bridsons-algorithm-n-for-poisson-disc-sampling/
    // All allocations are removed and the caller is responsible for allocating memory of the respective size
    // Use the upper bound methods from the parameter struct to allocate accordingly
    // The function returns the number of found points which are stored in the resultingPoints
    uint32_t maximalPoissonDiskSampling(
        std::span<Point> resultingPoints,
        std::span<int32_t> workingBufferGrid,
        std::span<int32_t> workingBufferActiveSet,
        const Parameters& parameters);
}