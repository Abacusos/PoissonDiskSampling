#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "PoissonDiskSampling.h"

#include <cassert>

namespace {

	std::tuple< std::vector<pds::Point>, std::vector<int32_t>, std::vector<int32_t>> generateTestData(const pds::Parameters& parameters) {
		uint32_t maximumCellCount = parameters.getMaximumCellCount();
		uint32_t maximumResultCount = parameters.getMaximumResultCount();

		std::vector<pds::Point> points(maximumResultCount);
		std::vector<int32_t> gridBuffer(maximumCellCount);
		std::vector<int32_t> activeBuffer(maximumCellCount);

		return { std::move(points) , std::move(gridBuffer) , std::move(activeBuffer) };
	}
}

TEST_CASE("Fast Poisson Disk Sampling generates no points for invalid input")
{
	pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 100.0f, .seed = 42 };

	auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);

	auto resultingPointCount = pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters);

	REQUIRE(resultingPointCount == 0);
}

TEST_CASE("Maximal Poisson Disk Sampling generates no points for invalid input")
{
	pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 100.0f, .seed = 42 };

	auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);

	auto resultingPointCount = pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters);

	REQUIRE(resultingPointCount == 0);
}
TEST_CASE("Fast Poisson Disk Sampling generates points with at least the defined radius")
{
	pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 100.0f, .seed = 42 };

	auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);

	auto resultingPointCount = pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters);

	REQUIRE(resultingPointCount == 0);
}

TEST_CASE("Maximal Poisson Disk Sampling generates points with at least the defined radius")
{
	pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 0.75f, .seed = 42 };

	auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);

	auto resultingPointCount = pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters);

	for (auto i = 0u; i < resultingPointCount; ++i) {
		for (auto j = 0u; j < resultingPointCount; ++j) {
			if (i == j) {
				continue;
			}
			REQUIRE(std::sqrt(std::pow((points[i].x - points[j].x), 2u) + std::pow((points[i].y - points[j].y), 2u)) > parameters.minimumDistance);
		}
	}
}

TEST_CASE("MaximalPoissonDiskSampling generates points with at least the defined radius")
{
	pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 0.75f, .seed = 42 };

	auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);

	auto resultingPointCount = pds::maximalPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters);

	for (auto i = 0u; i < resultingPointCount; ++i) {
		for (auto j = 0u; j < resultingPointCount; ++j) {
			if (i == j) {
				continue;
			}
			REQUIRE(std::sqrt(std::pow((points[i].x - points[j].x), 2u) + std::pow((points[i].y - points[j].y), 2u)) > parameters.minimumDistance);
		}
	}
}


TEST_CASE("PDS Benchmarks")
{
	BENCHMARK_ADVANCED("Fast Poisson Disk Sampling - 10x10 0.5")(Catch::Benchmark::Chronometer meter)
	{
		pds::Parameters parameters{ .xMaximum = 10.0f, .yMaximum = 10.0f, .minimumDistance = 0.5f, .seed = 42 };
		auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);
		meter.measure([&] { return pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters); });
	};

	BENCHMARK_ADVANCED("Maximal Poisson Disk Sampling - 10x10 0.5")(Catch::Benchmark::Chronometer meter)
	{
		pds::Parameters parameters{ .xMaximum = 10.0f, .yMaximum = 10.0f, .minimumDistance = 0.5f, .seed = 42 };
		auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);
		meter.measure([&] { return pds::maximalPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters); });
	};

	BENCHMARK_ADVANCED("Fast Poisson Disk Sampling - 50x50 0.5")(Catch::Benchmark::Chronometer meter)
	{
		pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 0.5f, .seed = 42 };
		auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);
		meter.measure([&] { return pds::fastPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters); });
	};

	BENCHMARK_ADVANCED("Maximal Poisson Disk Sampling - 50x50 0.5")(Catch::Benchmark::Chronometer meter)
	{
		pds::Parameters parameters{ .xMaximum = 50.0f, .yMaximum = 50.0f, .minimumDistance = 0.5f, .seed = 42 };
		auto [points, gridBuffer, activeBuffer] = generateTestData(parameters);
		meter.measure([&] { return pds::maximalPoissonDiskSampling(points, gridBuffer, activeBuffer, parameters); });
	};
}