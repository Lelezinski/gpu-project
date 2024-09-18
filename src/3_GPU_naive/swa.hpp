#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "utils.hpp"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int seqLen, SWAParams params);
__global__ void smithWatermanKernel_P1(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params);
__global__ void smithWatermanKernel_P2(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params);
void printScoreMatrix(const std::vector<int> &scoreMatrix, const std::string &seqA, const std::string &seqB);
