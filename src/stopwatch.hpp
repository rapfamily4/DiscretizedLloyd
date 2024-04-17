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
		return std::chrono::duration<double, std::milli>(endTimeStamp - beginTimeStamp).count();
	}
};

#endif