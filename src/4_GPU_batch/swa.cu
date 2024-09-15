#include "swa.hpp"
#include <cuda_runtime.h>

// Kernel function to compute Smith-Waterman score matrices for multiple sequences in parallel
__global__ void smithWatermanBatchKernel(int* scoreMatrices, const char* query, const char* database, const int* seqOffsets, int queryLength, int* dbSeqLengths, int dbCount, int match, int mismatch, int gap_penalty) {
    int dbIndex = blockIdx.x; // Each block processes one database sequence
    if (dbIndex >= dbCount) return;

    int dbSeqLength = dbSeqLengths[dbIndex];
    int scoreIdx = dbIndex * (queryLength + 1) * (dbSeqLength + 1);
    int cols = dbSeqLength + 1;

    // Each thread processes one column of the current row in the score matrix
    for (int i = 1; i <= queryLength; ++i) {
        for (int j = threadIdx.x + 1; j <= dbSeqLength; j += blockDim.x) {
            int matchScore = (query[i - 1] == database[seqOffsets[dbIndex] + j - 1]) ? match : mismatch;
            int scoreDiag = scoreMatrices[scoreIdx + (i - 1) * cols + (j - 1)] + matchScore;
            int scoreUp = scoreMatrices[scoreIdx + (i - 1) * cols + j] + gap_penalty;
            int scoreLeft = scoreMatrices[scoreIdx + i * cols + (j - 1)] + gap_penalty;

            scoreMatrices[scoreIdx + i * cols + j] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));
        }
        __syncthreads();
    }
}

// Host function for running the Smith-Waterman batch alignment
std::vector<SWAResult> smithWatermanBatch(const std::string &query, const std::vector<std::string> &database, int match, int mismatch, int gap_penalty) {
    int queryLength = query.size();
    int dbCount = database.size();

    // Precompute lengths of database sequences
    std::vector<int> dbSeqLengths(dbCount);
    std::vector<int> seqOffsets(dbCount + 1, 0);
    int totalDBSize = 0;

    for (int i = 0; i < dbCount; ++i) {
        dbSeqLengths[i] = database[i].size();
        seqOffsets[i + 1] = seqOffsets[i] + database[i].size();
        totalDBSize += database[i].size();
    }

    // Allocate memory for database sequences concatenated into a single array
    char* d_database;
    cudaMalloc(&d_database, totalDBSize * sizeof(char));
    
    // Copy database sequences to the device
    std::string flattenedDB;
    for (const auto& seq : database) {
        flattenedDB += seq;
    }
    cudaMemcpy(d_database, flattenedDB.c_str(), totalDBSize * sizeof(char), cudaMemcpyHostToDevice);

    // Allocate memory for the query sequence on the device
    char* d_query;
    cudaMalloc(&d_query, queryLength * sizeof(char));
    cudaMemcpy(d_query, query.c_str(), queryLength * sizeof(char), cudaMemcpyHostToDevice);

    // Allocate memory for sequence lengths and offsets
    int* d_dbSeqLengths;
    cudaMalloc(&d_dbSeqLengths, dbCount * sizeof(int));
    cudaMemcpy(d_dbSeqLengths, dbSeqLengths.data(), dbCount * sizeof(int), cudaMemcpyHostToDevice);

    int* d_seqOffsets;
    cudaMalloc(&d_seqOffsets, (dbCount + 1) * sizeof(int));
    cudaMemcpy(d_seqOffsets, seqOffsets.data(), (dbCount + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for score matrices (one per database sequence)
    int totalMatrixSize = 0;
    for (int i = 0; i < dbCount; ++i) {
        totalMatrixSize += (queryLength + 1) * (dbSeqLengths[i] + 1);
    }
    int* d_scoreMatrices;
    cudaMalloc(&d_scoreMatrices, totalMatrixSize * sizeof(int));
    cudaMemset(d_scoreMatrices, 0, totalMatrixSize * sizeof(int));  // Initialize all scores to zero

    // Launch the kernel to compute the score matrices in parallel
    int threadsPerBlock = 256;
    int blocksPerGrid = dbCount;
    smithWatermanBatchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_scoreMatrices, d_query, d_database, d_seqOffsets, queryLength, d_dbSeqLengths, dbCount, match, mismatch, gap_penalty);

    // Copy back the result matrices to the host
    std::vector<int> h_scoreMatrices(totalMatrixSize);
    cudaMemcpy(h_scoreMatrices.data(), d_scoreMatrices, totalMatrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Traceback and generate the alignment for each sequence
    std::vector<SWAResult> results;
    for (int dbIndex = 0; dbIndex < dbCount; ++dbIndex) {
        int maxScore = 0;
        int maxRow = 0, maxCol = 0;
        int cols = dbSeqLengths[dbIndex] + 1;
        int scoreIdx = dbIndex * (queryLength + 1) * cols;

        // Find the maximum score and its position in the matrix
        for (int i = 1; i <= queryLength; ++i) {
            for (int j = 1; j <= dbSeqLengths[dbIndex]; ++j) {
                if (h_scoreMatrices[scoreIdx + i * cols + j] > maxScore) {
                    maxScore = h_scoreMatrices[scoreIdx + i * cols + j];
                    maxRow = i;
                    maxCol = j;
                }
            }
        }

        // Perform traceback (on the CPU)
        std::string alignedSeqA, alignedSeqB;
        int i = maxRow, j = maxCol;
        while (i > 0 && j > 0 && h_scoreMatrices[scoreIdx + i * cols + j] != 0) {
            int currentScore = h_scoreMatrices[scoreIdx + i * cols + j];
            int diagScore = h_scoreMatrices[scoreIdx + (i - 1) * cols + (j - 1)];
            int upScore = h_scoreMatrices[scoreIdx + (i - 1) * cols + j];
            int leftScore = h_scoreMatrices[scoreIdx + i * cols + (j - 1)];

            if (currentScore == diagScore + ((query[i - 1] == database[seqOffsets[dbIndex] + j - 1]) ? match : mismatch)) {
                alignedSeqA = query[i - 1] + alignedSeqA;
                alignedSeqB = database[seqOffsets[dbIndex] + j - 1] + alignedSeqB;
                i--;
                j--;
            } else if (currentScore == upScore + gap_penalty) {
                alignedSeqA = query[i - 1] + alignedSeqA;
                alignedSeqB = '-' + alignedSeqB;
                i--;
            } else if (currentScore == leftScore + gap_penalty) {
                alignedSeqA = '-' + alignedSeqA;
                alignedSeqB = database[seqOffsets[dbIndex] + j - 1] + alignedSeqB;
                j--;
            }
        }

        results.push_back({alignedSeqA, alignedSeqB, maxScore});
    }

    // Free device memory
    cudaFree(d_scoreMatrices);
    cudaFree(d_query);
    cudaFree(d_database);
    cudaFree(d_dbSeqLengths);
    cudaFree(d_seqOffsets);

    return results;
}

// Print the score matrix
void printScoreMatrix(const std::vector<std::vector<int>> &scoreMatrix, const std::string &seqA, const std::string &seqB)
{
    std::cout << "Score Matrix:\n  ";
    for (char b : seqB)
    {
        std::cout << "  " << b;
    }
    std::cout << "\n";
    for (int i = 0; i < scoreMatrix.size(); ++i)
    {
        if (i > 0)
            std::cout << seqA[i - 1] << " ";
        else
            std::cout << "  ";
        for (int j = 0; j < scoreMatrix[0].size(); ++j)
        {
            std::cout << scoreMatrix[i][j] << "  ";
        }
        std::cout << "\n";
    }
}

// Print the traceback path
void printTracebackPath(const std::vector<std::vector<int>> &traceback, const std::string &seqA, const std::string &seqB)
{
    std::cout << "Traceback Path:\n  ";
    for (char b : seqB)
    {
        std::cout << "  " << b;
    }
    std::cout << "\n";
    for (int i = 0; i < traceback.size(); ++i)
    {
        if (i > 0)
            std::cout << seqA[i - 1] << " ";
        else
            std::cout << "  ";
        for (int j = 0; j < traceback[0].size(); ++j)
        {
            std::cout << traceback[i][j] << "  ";
        }
        std::cout << "\n";
    }
}
