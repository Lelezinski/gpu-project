#include "swa.hpp"

// Kernel function to compute the score matrix in parallel
__global__ void smithWatermanKernel_P1(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params)
{
    int i = threadIdx.x + 1;       // Col index
    int j = (antidiagDim - i) + 1; // Row index

    // Match or mismatch?
    int matchScore = (seqA[j - 1] == seqB[i - 1]) ? params.match : params.mismatch;
    int scoreDiag = scoreMatrix[(j - 1) * scoreMatrixDim + (i - 1)] + matchScore;
    // Gap from North
    int scoreUp = scoreMatrix[(j - 1) * scoreMatrixDim + i] + params.gap;
    // Gap from West
    int scoreLeft = scoreMatrix[j * scoreMatrixDim + (i - 1)] + params.gap;

    // scoreMatrix[j * cols + i] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));

    // TEST TO CHECK ANTIDIAGONAL
    scoreMatrix[j * scoreMatrixDim + i] = j;
}

__global__ void smithWatermanKernel_P2(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params)
{
    int i = threadIdx.x + antidiagDim - scoreMatrixDim; // Col index
    int j = (antidiagDim - i) + 1;                      // Row index

    // Match or mismatch?
    int matchScore = (seqA[j - 1] == seqB[i - 1]) ? params.match : params.mismatch;
    int scoreDiag = scoreMatrix[(j - 1) * scoreMatrixDim + (i - 1)] + matchScore;
    // Gap from North
    int scoreUp = scoreMatrix[(j - 1) * scoreMatrixDim + i] + params.gap;
    // Gap from West
    int scoreLeft = scoreMatrix[j * scoreMatrixDim + (i - 1)] + params.gap;

    // scoreMatrix[j * cols + i] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));

    // TEST TO CHECK ANTIDIAGONAL
    scoreMatrix[j * scoreMatrixDim + i] = j;
}

// Host function
SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int seqLen, SWAParams params)
{
    int i = 1;

    // Account for first row-col
    int scoreMatrixDim = seqLen + 1;
    size_t scoreMatrixSize = scoreMatrixDim * scoreMatrixDim * sizeof(int);

    // Allocate memory on the host using vector
    std::vector<int> h_scoreMatrix(scoreMatrixDim * scoreMatrixDim, 0);

    // Allocate memory on the device
    int *d_scoreMatrix;
    cudaMalloc(&d_scoreMatrix, scoreMatrixSize);

    char *d_seqA;
    char *d_seqB;
    cudaMalloc(&d_seqA, seqLen * sizeof(char));
    cudaMalloc(&d_seqB, seqLen * sizeof(char));

    // Copy sequences to the device
    cudaMemcpy(d_seqA, seqA.c_str(), seqLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seqB, seqB.c_str(), seqLen * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize the matrix to zero on the device
    cudaMemcpy(d_scoreMatrix, h_scoreMatrix.data(), scoreMatrixSize, cudaMemcpyHostToDevice);

    /* ------------------------- Launch the CUDA kernel ------------------------- */

    TimePoint start = std::chrono::high_resolution_clock::now();

    // Start from top left corner
    for (i = 1; i < scoreMatrixDim + 1; i++)
    {
        // Move towards the antidiagonal
        smithWatermanKernel_P1<<<1, i>>>(d_scoreMatrix, d_seqA, d_seqB, i, scoreMatrixDim, params);
    }

    // // Continue to other corner
    // for (int i = shortestEdge; i < longestEdge; i++)
    // {
    //     smithWatermanKernel<<<1, shortestEdge - 1>>>(d_scoreMatrix, d_seqA, d_seqB, shortestEdge - 1, i, rows, cols, match, mismatch, gap_penalty);
    // }

    // Reach bottom right
    // for (int k = scoreMatrixDim - 1; k > 0; k--)
    // {
    //     smithWatermanKernel2<<<1, k>>>(d_scoreMatrix, d_seqA, d_seqB, k, cols, params);
    //     i++;
    // }

    // Copy the score matrix back to the host
    cudaMemcpy(h_scoreMatrix.data(), d_scoreMatrix, scoreMatrixSize, cudaMemcpyDeviceToHost);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_scoreMatrix);
        cudaFree(d_seqA);
        cudaFree(d_seqB);
        return {"", "", -1};
    }

    // Find the maximum score and its position in the matrix (traceback)
    int maxScore = 0;
    int maxRow = 0, maxCol = 0;
    for (int i = 1; i < scoreMatrixDim; ++i)
    {
        for (int j = 1; j < scoreMatrixDim; ++j)
        {
            if (h_scoreMatrix[i * scoreMatrixDim + j] > maxScore)
            {
                maxScore = h_scoreMatrix[i * scoreMatrixDim + j];
                maxRow = i;
                maxCol = j;
            }
        }
    }

    TimePoint end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "SW Kernel executed in " << elapsed.count() << " seconds.\n";

    printScoreMatrix(h_scoreMatrix, seqA, seqB);

    // Perform traceback on the host
    std::string alignedSeqA, alignedSeqB;
    int r = maxRow, c = maxCol;
    while (false && r > 0 && c > 0 && h_scoreMatrix[r * scoreMatrixDim + c] != 0)
    {
        int currentScore = h_scoreMatrix[r * scoreMatrixDim + c];
        std::cout << "TRACEBACK: [" << r << "][" << c << "] - Current Score: " << currentScore << "\n";
        int diagScore = h_scoreMatrix[(r - 1) * scoreMatrixDim + (c - 1)];
        int upScore = h_scoreMatrix[(r - 1) * scoreMatrixDim + c];
        int leftScore = h_scoreMatrix[r * scoreMatrixDim + (c - 1)];

        if (currentScore == diagScore + ((seqA[r - 1] == seqB[c - 1]) ? params.match : params.mismatch))
        {
            alignedSeqA = seqA[r - 1] + alignedSeqA;
            alignedSeqB = seqB[c - 1] + alignedSeqB;
            r--;
            c--;
        }
        else if (currentScore == upScore + params.gap)
        {
            alignedSeqA = seqA[r - 1] + alignedSeqA;
            alignedSeqB = '-' + alignedSeqB;
            r--;
        }
        else if (currentScore == leftScore + params.gap)
        {
            alignedSeqA = '-' + alignedSeqA;
            alignedSeqB = seqB[c - 1] + alignedSeqB;
            c--;
        }
    }

    // Free device memory
    cudaFree(d_scoreMatrix);
    cudaFree(d_seqA);
    cudaFree(d_seqB);

    return {alignedSeqA, alignedSeqB, maxScore};
}

// Print the score matrix
void printScoreMatrix(const std::vector<int> &scoreMatrix, const std::string &seqA, const std::string &seqB)
{
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    std::cout << "Score Matrix:\n   ";
    for (char b : seqB)
    {
        std::cout << "\t" << b;
    }
    std::cout << "\n";
    for (int i = 0; i < rows; ++i)
    {
        if (i > 0)
            std::cout << seqA[i - 1] << "\t";
        else
            std::cout << "\t";
        for (int j = 0; j < cols; ++j)
        {
            std::cout << scoreMatrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}
