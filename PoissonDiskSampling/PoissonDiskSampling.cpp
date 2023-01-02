#include "PoissonDiskSampling.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>

namespace pds {

	inline uint32_t Parameters::getMaximumCellCount() const
	{
		const auto cellSize = minimumDistance / std::sqrt(pds::numberDimensions);
		const auto xMaximumCell = static_cast<std::uint32_t>(std::ceil(xMaximum / cellSize));
		const auto yMaximumCell = static_cast<std::uint32_t>(std::ceil(yMaximum / cellSize));
		return xMaximumCell * yMaximumCell;
	}
	inline uint32_t Parameters::getMaximumResultCount() const
	{
		const auto xMaximumResult = static_cast<std::uint32_t>(std::ceil(xMaximum / minimumDistance));
		const auto yMaximumResult = static_cast<std::uint32_t>(std::ceil(yMaximum / minimumDistance));
		return xMaximumResult * yMaximumResult;
	}

	namespace {
		// This corresponds to the k chosen in the paper
		constexpr std::uint32_t rejectionSampleCount = 30;

		// Simple implementation of a uniform grid to store the used indices per cell
		class Grid {
		private:
			constexpr static int32_t invalidIndex = -1;
		public:

			explicit Grid(std::span<int32_t> grid, std::span<const Point> resultingPoints, std::uint32_t sizeX, std::uint32_t sizeY, float minimumDistance, float cellSize) :
				m_grid{ grid },
				m_resultingPoints{ resultingPoints },
				m_sizeX{ sizeX }, m_sizeY{ sizeY },
				m_squaredMinimumDistance{ minimumDistance * minimumDistance },
				m_cellSize{ cellSize } {
				std::fill_n(grid.begin(), grid.size(), invalidIndex);
			};

			Grid() = delete;
			~Grid() = default;
			Grid(const Grid&) = delete;
			Grid(Grid&&) = delete;
			Grid& operator=(const Grid&) = delete;
			Grid& operator=(Grid&&) = delete;

			bool pointFree(const Point& point) {
				std::uint32_t centerX = static_cast<std::uint32_t>(point.x / m_cellSize);
				std::uint32_t centerY = static_cast<std::uint32_t>(point.y / m_cellSize);

				// We need to search through 2 neighbours in each direction because the cellsize was chosen in such a way
				// that we have at most one point in each cell. To guarantee that in 2D, we need to look at the worst case
				// which is a point in a corner. The next possible point which must be at least one cell away is at the opposite
				// corner of the cell, so we are looking for a cell size where the diagonale is at most the radius, thus the 
				// cell size is radius / sqrt(2) which is smaller than radius.
				// But for this function we are searching for potential neighboring cells so we need to look in all potential cells.
				// These can only be in the next two neighbors in each direction based on the reasoning from above. 
				const std::uint32_t minX = centerX > 2 ? centerX - 2 : 0;
				const std::uint32_t minY = centerY > 2 ? centerY - 2 : 0;

				const std::uint32_t maxX = std::min(centerX + 3, m_sizeX);
				const std::uint32_t maxY = std::min(centerY + 3, m_sizeY);

				for (std::uint32_t y = minY; y < maxY; ++y)
				{
					uint32_t rowOffset = y * m_sizeX;
					for (std::uint32_t x = minX; x < maxX; ++x)
					{
						auto index = m_grid[rowOffset + x];
						if (index != invalidIndex)
						{
							auto distanceSquared = std::pow(point.x - m_resultingPoints[index].x, 2u) + std::pow(point.y - m_resultingPoints[index].y, 2u);
							if (distanceSquared < m_squaredMinimumDistance)
							{
								return false;
							}
						}
					}
				}
				return true;
			};

			void insertPoint(const Point& point, uint32_t index) {
				std::uint32_t centerX = static_cast<std::uint32_t>(point.x / m_cellSize);
				std::uint32_t centerY = static_cast<std::uint32_t>(point.y / m_cellSize);

				m_grid[centerY * m_sizeX + centerX] = index;
			}

		private:

			std::span<int32_t> m_grid;
			std::span<const Point> m_resultingPoints;

			std::uint32_t m_sizeX;
			std::uint32_t m_sizeY;
			float m_squaredMinimumDistance;
			float m_cellSize;
		};

		// This function mostly corresponds to step 1 from the mentioned paper
		void insertInitialPoint(std::span<Point> resultingPoints, std::span<int32_t> workingBufferActiveSet, const Parameters& parameters, Grid& grid, std::default_random_engine& engine)
		{
			std::uniform_real_distribution<float> xDistribution(0.0f, parameters.xMaximum);
			std::uniform_real_distribution<float> yDistribution(0.0f, parameters.yMaximum);
			resultingPoints[0] = Point{ .x = xDistribution(engine), .y = yDistribution(engine) };
			grid.insertPoint(resultingPoints[0], 0);
			workingBufferActiveSet[0] = 0;
		}
	}


	std::uint32_t fastPoissonDiskSampling(std::span<Point> resultingPoints, std::span<int32_t> workingBufferGrid, std::span<int32_t> workingBufferActiveSet, const Parameters& parameters)
	{
		const auto cellSize = parameters.minimumDistance / static_cast<float>(std::sqrt(numberDimensions));

		const auto xMaximumCell = static_cast<std::uint32_t>(std::ceil(parameters.xMaximum / cellSize));
		const auto yMaximumCell = static_cast<std::uint32_t>(std::ceil(parameters.yMaximum / cellSize));

		if (!(parameters.minimumDistance > 0.0f && parameters.xMaximum > 0.0f && parameters.yMaximum > 0.0f
			&& parameters.xMaximum > parameters.minimumDistance && parameters.yMaximum > parameters.minimumDistance
			&& resultingPoints.size() >= parameters.getMaximumResultCount()
			&& workingBufferGrid.size() >= parameters.getMaximumCellCount()
			&& workingBufferActiveSet.size() >= parameters.getMaximumResultCount()))
		{
			return 0;
		}

		Grid grid{ workingBufferGrid, resultingPoints, xMaximumCell, yMaximumCell, parameters.minimumDistance, cellSize };

		std::default_random_engine engine(parameters.seed);

		insertInitialPoint(resultingPoints, workingBufferActiveSet, parameters, grid, engine);
		std::uint32_t foundPoints = 1;
		// The paper says to randomly choose a point from the active set. For my purposes, I decided to instead use
		// a stack on the working buffer active set instead to remove potential allocations
		std::uint32_t activePointsWriteIndex = 1;

		std::uniform_real_distribution<float> thetaDistribution(0.0f, static_cast<float>(std::numbers::pi) * 2.f);
		std::uniform_real_distribution<float> radiusDistribution(parameters.minimumDistance * parameters.minimumDistance, parameters.minimumDistance);

		while (activePointsWriteIndex != 0)
		{
			std::uint32_t activeIndex = workingBufferActiveSet[--activePointsWriteIndex];
			Point startingPoint = resultingPoints[activeIndex];

			for (auto i = 0u; i < rejectionSampleCount; ++i)
			{
				// This picks a point uniformly from the spherical annulus 
				// although this is maybe not recommended depending on what one want to achieve
				// picking closer to the inner radius will lead to a denser distribution
				float theta = thetaDistribution(engine);
				float radius = std::sqrt(radiusDistribution(engine));
				Point point = { .x = startingPoint.x + radius * std::cos(theta), .y = startingPoint.y + radius * std::sin(theta) };

				if (point.x < 0.0f || point.x > parameters.xMaximum || point.y < 0.0f || point.y > parameters.yMaximum) {
					continue;
				}

				if (grid.pointFree(point)) {
					resultingPoints[foundPoints] = point;
					grid.insertPoint(point, foundPoints);
					workingBufferActiveSet[activePointsWriteIndex] = foundPoints;
					++foundPoints;
					++activePointsWriteIndex;
				}
			}
		}

		return foundPoints;
	}
	uint32_t maximalPoissonDiskSampling(std::span<Point> resultingPoints, std::span<int32_t> workingBufferGrid, std::span<int32_t> workingBufferActiveSet, const Parameters& parameters)
	{
		const auto cellSize = parameters.minimumDistance / static_cast<float>(std::sqrt(numberDimensions));

		const auto xMaximumCell = static_cast<std::uint32_t>(std::ceil(parameters.xMaximum / cellSize));
		const auto yMaximumCell = static_cast<std::uint32_t>(std::ceil(parameters.yMaximum / cellSize));

		if (!(parameters.minimumDistance > 0.0f && parameters.xMaximum > 0.0f && parameters.yMaximum > 0.0f
			&& parameters.xMaximum > parameters.minimumDistance && parameters.yMaximum > parameters.minimumDistance
			&& resultingPoints.size() >= parameters.getMaximumResultCount()
			&& workingBufferGrid.size() >= parameters.getMaximumCellCount()
			&& workingBufferActiveSet.size() >= parameters.getMaximumResultCount()))
		{
			return 0;
		}

		Grid grid{ workingBufferGrid, resultingPoints, xMaximumCell, yMaximumCell, parameters.minimumDistance, cellSize };

		std::default_random_engine engine(parameters.seed);

		insertInitialPoint(resultingPoints, workingBufferActiveSet, parameters, grid, engine);
		std::uint32_t foundPoints = 1;
		std::uint32_t activePointsWriteIndex = 1;

		std::uniform_real_distribution<float> seedDistribution(0.0f, 1.0f);
		float epsilon = 0.0000001f;
		float twoPi = 2.0f * static_cast<float>(std::numbers::pi);

		while (activePointsWriteIndex != 0)
		{
			std::uint32_t activeIndex = workingBufferActiveSet[--activePointsWriteIndex];
			Point startingPoint = resultingPoints[activeIndex];

			float seed = seedDistribution(engine);

			for (auto i = 0u; i < rejectionSampleCount; ++i)
			{
				// This does not sample uniformly but the auther claims that it does not matter
				// It should generate denser point sets because points are chosen on the circle of 
				// minimumDistance only instead of between minimumDistance and 2 times minimumDistance
				float theta = twoPi * (seed + (i / static_cast<float>(rejectionSampleCount)));
				float radius = parameters.minimumDistance + epsilon;
				Point point = { .x = startingPoint.x + radius * std::cos(theta), .y = startingPoint.y + radius * std::sin(theta) };

				if (point.x < 0.0f || point.x > parameters.xMaximum || point.y < 0.0f || point.y > parameters.yMaximum) {
					continue;
				}

				if (grid.pointFree(point)) {
					resultingPoints[foundPoints] = point;
					grid.insertPoint(point, foundPoints);
					workingBufferActiveSet[activePointsWriteIndex] = foundPoints;
					++foundPoints;
					++activePointsWriteIndex;
				}
			}
		}

		return foundPoints;
	}
}