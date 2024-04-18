#ifndef PERFORMANCE_STATS_H
#define PERFORMANCE_STATS_H

#include "consts.hpp"

class PerformanceStatistics {
private:
	unsigned int iterationsCount;
	double minTime;
	double maxTime;
	double totTime;

public:
	PerformanceStatistics() :
		iterationsCount{ 0 },
		minTime{ +INF },
		maxTime{ -INF },
		totTime{ 0.0 } {}

	void reset() {
		iterationsCount = 0;
		minTime = +INF;
		maxTime = -INF;
		totTime = 0.0;
	}

	void recordIteration(double timeInMillis) {
		if (minTime > timeInMillis)
			minTime = timeInMillis;
		if (maxTime < timeInMillis)
			maxTime = timeInMillis;
		totTime += timeInMillis;
		iterationsCount++;
	}

	unsigned int getIterations() const { return iterationsCount; }
	double getMinTime() const { return minTime; }
	double getMaxTime() const { return maxTime; }
	double getAvgTime() const { 
		if (iterationsCount > 0)
			return totTime / iterationsCount;
		return 0.0;
	}
};

#endif