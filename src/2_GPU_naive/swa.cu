#include "swa.hpp"

// Kernel functions to compute the score matrix in parallel
__global__ void smithWatermanKernel_P1(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x + 1; // Global thread ID
    int i = thread_id;                                         // Col index
    int j = (antidiagDim - i) + 1;                             // Row index

    if (i <= antidiagDim && j > 0 && j <= scoreMatrixDim)
    {
        // Match or mismatch?
        int matchScore = (seqA[j - 1] == seqB[i - 1]) ? params.match : params.mismatch;
        int scoreDiag = scoreMatrix[(j - 1) * scoreMatrixDim + (i - 1)] + matchScore;
        // Gap from North
        int scoreUp = scoreMatrix[(j - 1) * scoreMatrixDim + i] + params.gap;
        // Gap from West
        int scoreLeft = scoreMatrix[j * scoreMatrixDim + (i - 1)] + params.gap;

        scoreMatrix[j * scoreMatrixDim + i] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));
    }
}

__global__ void smithWatermanKernel_P2(int *scoreMatrix, const char *seqA, const char *seqB, const int antidiagDim, const int scoreMatrixDim, const SWAParams params)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x + 1; // Global thread ID
    int i = thread_id + antidiagDim - scoreMatrixDim;          // Col index
    int j = (antidiagDim - i) + 1;                             // Row index

    if (i > 0 && i <= scoreMatrixDim && j > 0 && j <= scoreMatrixDim)
    {
        // Match or mismatch?
        int matchScore = (seqA[j - 1] == seqB[i - 1]) ? params.match : params.mismatch;
        int scoreDiag = scoreMatrix[(j - 1) * scoreMatrixDim + (i - 1)] + matchScore;
        // Gap from North
        int scoreUp = scoreMatrix[(j - 1) * scoreMatrixDim + i] + params.gap;
        // Gap from West
        int scoreLeft = scoreMatrix[j * scoreMatrixDim + (i - 1)] + params.gap;

        scoreMatrix[j * scoreMatrixDim + i] = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));
    }
}

// Host function
SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int seqLen, SWAParams params)
{
    int scoreMatrixDim = seqLen + 1; // Account for first row-col
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

    int threadsPerBlock = 1024; // Max threads per block for CUDA
    int blocks, numThreads, ii;

    TimePoint start = std::chrono::high_resolution_clock::now();

    // Phase 1: From top-left to middle
    for (ii = 1; ii < scoreMatrixDim; ii++)
    {
        numThreads = min(ii, threadsPerBlock);
        blocks = (ii + numThreads - 1) / numThreads;
        smithWatermanKernel_P1<<<blocks, numThreads>>>(d_scoreMatrix, d_seqA, d_seqB, ii, scoreMatrixDim, params);
    }

    // Phase 2: From middle to bottom-right
    for (int jj = scoreMatrixDim; jj > 0; jj--)
    {
        numThreads = min(jj, threadsPerBlock);
        blocks = (jj + numThreads - 1) / numThreads;
        smithWatermanKernel_P2<<<blocks, numThreads>>>(d_scoreMatrix, d_seqA, d_seqB, ii, scoreMatrixDim, params);
        ii++;
    }

    // Copy the score matrix back to the host
    cudaMemcpy(h_scoreMatrix.data(), d_scoreMatrix, scoreMatrixSize, cudaMemcpyDeviceToHost);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

        cudaFree(d_scoreMatrix);
        cudaFree(d_seqA);
        cudaFree(d_seqB);
        return {"", "", -1};
    }

    TimePoint end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "SW Kernel executed in " << elapsed.count() << " seconds. Cell updates per second (CUPS): " << ((scoreMatrixDim * scoreMatrixDim) / elapsed.count()) << " \n";

    // DEBUG
    //printScoreMatrix(h_scoreMatrix, seqA, seqB);

    /* -------------------------------- Traceback ------------------------------- */

    int maxScore = 0;
    int maxRow = 0, maxCol = 0;

    // Find highest score value
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

    // DEBUG
    //printf("Max element at: [%d, %d]\n", maxRow, maxCol);

    std::string alignedSeqA, alignedSeqB;
    int i = maxRow, j = maxCol;

    while (i > 0 && j > 0 && h_scoreMatrix[i * scoreMatrixDim + j] != 0)
    {
        int currentScore = h_scoreMatrix[i * scoreMatrixDim + j];
        // std::cout << "TRACEBACK: [" << i << "][" << j << "] - Current Score: " << currentScore << "\n";
        int diagScore = h_scoreMatrix[(i - 1) * scoreMatrixDim + (j - 1)];
        int upScore = h_scoreMatrix[(i - 1) * scoreMatrixDim + j];
        int leftScore = h_scoreMatrix[i * scoreMatrixDim + (j - 1)];

        if (currentScore == diagScore + params.match || currentScore == diagScore + params.mismatch)
        {
            // Move diagonal
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            i--;
            j--;
        }
        else if (currentScore == upScore + params.gap)
        {
            // Move up
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = '-' + alignedSeqB;
            i--;
        }
        else
        {
            // Move left
            alignedSeqA = '-' + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            j--;
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

    std::cout << "Score Matrix:\n\t";
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
