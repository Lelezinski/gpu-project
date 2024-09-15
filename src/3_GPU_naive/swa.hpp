#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct SWAResult
{
    std::string alignedSeqA;
    std::string alignedSeqB;
    int score;
};

// Host function declaration
SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, int match, int mismatch, int gap_penalty);
__global__ void smithWatermanKernel(int *scoreMatrix, const char *seqA, const char *seqB, int rows, int cols, int match, int mismatch, int gap_penalty);
void printScoreMatrix(const std::vector<std::vector<int>> &scoreMatrix, const std::string &seqA, const std::string &seqB);
void printTracebackPath(const std::vector<std::vector<int>> &traceback, const std::string &seqA, const std::string &seqB);