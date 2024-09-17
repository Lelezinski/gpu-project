#include "swa.hpp"
#include <cuda_runtime.h>

// Kernel function to compute the Smith-Waterman score matrix in parallel
__global__ void smithWatermanKernel(int *scoreMatrix, const char *seqA, const char *seqB, int rows, int cols, int match, int mismatch, int gap_penalty)
{
    int i = blockIdx.x + 1;  // Each block processes a row
    int j = threadIdx.x + 1; // Each thread processes a column

    if (i < rows && j < cols)
    {
        int matchScore = (seqA[i - 1] == seqB[j - 1]) ? match : mismatch;
        int scoreDiag = scoreMatrix[(i - 1) * cols + (j - 1)] + matchScore;
        int scoreUp = scoreMatrix[(i - 1) * cols + j] + gap_penalty;
        int scoreLeft = scoreMatrix[i * cols + (j - 1)] + gap_penalty;

        // Compute the maximum score for the current cell
        scoreMatrix[i * cols + j] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));
    }
}

// Host function for Smith-Waterman algorithm
SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int match, const int mismatch, const int gap_penalty)
{
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;
    size_t matrixSize = rows * cols * sizeof(int);

    // Allocate memory on the host
    int *h_scoreMatrix = new int[rows * cols]();

    // Allocate memory on the device
    int *d_scoreMatrix;
    cudaMalloc(&d_scoreMatrix, matrixSize);

    char *d_seqA;
    char *d_seqB;
    cudaMalloc(&d_seqA, seqA.size() * sizeof(char));
    cudaMalloc(&d_seqB, seqB.size() * sizeof(char));

    // Copy sequences to the device
    cudaMemcpy(d_seqA, seqA.c_str(), seqA.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seqB, seqB.c_str(), seqB.size() * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize the matrix to zero on the device
    cudaMemcpy(d_scoreMatrix, h_scoreMatrix, matrixSize, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to fill the score matrix
    int numBlocks = rows - 1;
    int threadsPerBlock = cols - 1;
    smithWatermanKernel<<<numBlocks, threadsPerBlock>>>(d_scoreMatrix, d_seqA, d_seqB, rows, cols, match, mismatch, gap_penalty);

    // Copy the score matrix back to the host
    cudaMemcpy(h_scoreMatrix, d_scoreMatrix, matrixSize, cudaMemcpyDeviceToHost);

    // Find the maximum score and its position in the matrix (traceback)
    int maxScore = 0;
    int maxRow = 0, maxCol = 0;
    for (int i = 1; i < rows; ++i)
    {
        for (int j = 1; j < cols; ++j)
        {
            if (h_scoreMatrix[i * cols + j] > maxScore)
            {
                maxScore = h_scoreMatrix[i * cols + j];
                maxRow = i;
                maxCol = j;
            }
        }
    }

    // Perform traceback on the host
    std::string alignedSeqA, alignedSeqB;
    int i = maxRow, j = maxCol;
    while (i > 0 && j > 0 && h_scoreMatrix[i * cols + j] != 0)
    {
        int currentScore = h_scoreMatrix[i * cols + j];
        int diagScore = h_scoreMatrix[(i - 1) * cols + (j - 1)];
        int upScore = h_scoreMatrix[(i - 1) * cols + j];
        int leftScore = h_scoreMatrix[i * cols + (j - 1)];

        if (currentScore == diagScore + ((seqA[i - 1] == seqB[j - 1]) ? match : mismatch))
        {
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            i--;
            j--;
        }
        else if (currentScore == upScore + gap_penalty)
        {
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = '-' + alignedSeqB;
            i--;
        }
        else if (currentScore == leftScore + gap_penalty)
        {
            alignedSeqA = '-' + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            j--;
        }
    }

    // Free device memory
    cudaFree(d_scoreMatrix);
    cudaFree(d_seqA);
    cudaFree(d_seqB);
    delete[] h_scoreMatrix;

    return {alignedSeqA, alignedSeqB, maxScore};
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
