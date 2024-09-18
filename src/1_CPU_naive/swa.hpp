#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "utils.hpp"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int seqLen, SWAParams params);
void printScoreMatrix(const std::vector<std::vector<int>> &scoreMatrix, const std::string &seqA, const std::string &seqB);
