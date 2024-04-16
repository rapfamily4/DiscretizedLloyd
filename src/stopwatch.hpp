#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>

class Stopwatch {
private:
	std::chrono::steady_clock::time_point beginTimeStamp;
	std::chrono::steady_clock::time_point endTimeStamp;

public:
	Stopwatch() {}

	void begin() {
		beginTimeStamp = std::chrono::steady_clock::now();
	}

	double end() {
		endTimeStamp = std::chrono::steady_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(endTimeStamp - beginTimeStamp).count();
	}
};

#endif