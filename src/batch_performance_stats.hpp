#ifndef BATCH_PERFORMANCE_STATS_H
#define BATCH_PERFORMANCE_STATS_H

#include <vector>
#include <limits.h>
#include "consts.hpp"
#include "performance_stats.hpp"

class BatchPerformanceStatistics : public PerformanceStatistics {
private:
	std::vector<int> minPhaseCount;
	std::vector<int> maxPhaseCount;
	std::vector<int> totPhaseCount;

public:
	BatchPerformanceStatistics() {}

	void reset() override {
		PerformanceStatistics::reset();
		minPhaseCount.clear();
		maxPhaseCount.clear();
		totPhaseCount.clear();
	}

	void recordPhaseCounts(std::vector<int>& counts) {
		minPhaseCount.resize(counts.size(), INT_MAX);
		maxPhaseCount.resize(counts.size(), 0);
		totPhaseCount.resize(counts.size(), 0);
		assert(counts.size() == minPhaseCount.size() && counts.size() == maxPhaseCount.size() && counts.size() == totPhaseCount.size());
		for (int i = 0; i < counts.size(); i++) {
			if (minPhaseCount[i] > counts[i])
				minPhaseCount[i] = counts[i];
			if (maxPhaseCount[i] < counts[i])
				maxPhaseCount[i] = counts[i];
			totPhaseCount[i] += counts[i];
		}
		assert(maxPhaseCount.size() == minPhaseCount.size() && totPhaseCount.size() == maxPhaseCount.size() && minPhaseCount.size() == totPhaseCount.size());
	}

	int getMinPhaseCount(int phase) const {
		if (iterationsCount > 0 || (0 <= phase && phase < minPhaseCount.size()))
			return minPhaseCount[phase];
		return INT_MAX;
	}

	int getMaxPhaseCount(int phase) const {
		if (iterationsCount > 0 || (0 <= phase && phase < maxPhaseCount.size()))
			return maxPhaseCount[phase];
		return 0;
	}

	double getAvgPhaseCount(int phase) const {
		if (iterationsCount > 0)
			return totPhaseCount[phase] / iterationsCount;
		return 0.0;
	}
};

#endif