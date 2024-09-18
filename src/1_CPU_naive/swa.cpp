#include "swa.hpp"

// Smith-Waterman algorithm implementation
SWAResult smithWaterman(const std::string &seqA, const std::string &seqB, const int seqLen, SWAParams params)
{
    TimePoint start = std::chrono::high_resolution_clock::now();

    int scoreMatrixDim = seqLen + 1;

    // Score and traceback matrices
    std::vector<std::vector<int>> scoreMatrix(scoreMatrixDim, std::vector<int>(scoreMatrixDim, 0));
    std::vector<std::vector<int>> traceback(scoreMatrixDim, std::vector<int>(scoreMatrixDim, 0));

    int maxScore = 0;
    int maxRow = 0, maxCol = 0;

    // Fill the score and traceback matrices
    for (int i = 1; i < scoreMatrixDim; ++i)
    {
        for (int j = 1; j < scoreMatrixDim; ++j)
        {
            int matchScore = (seqA[i - 1] == seqB[j - 1]) ? params.match : params.mismatch;
            int scoreDiag = scoreMatrix[i - 1][j - 1] + matchScore;
            int scoreUp = scoreMatrix[i - 1][j] + params.gap;
            int scoreLeft = scoreMatrix[i][j - 1] + params.gap;

            // Calculate the maximum score
            scoreMatrix[i][j] = std::max(0, std::max(scoreDiag, std::max(scoreUp, scoreLeft)));

            // Track the highest score
            if (scoreMatrix[i][j] > maxScore)
            {
                maxScore = scoreMatrix[i][j];
                maxRow = i;
                maxCol = j;
            }

            // Track the traceback direction (1=Diagonal, 2=Up, 3=Left)
            if (scoreMatrix[i][j] == scoreDiag)
            {
                traceback[i][j] = 1;
            }
            else if (scoreMatrix[i][j] == scoreUp)
            {
                traceback[i][j] = 2;
            }
            else if (scoreMatrix[i][j] == scoreLeft)
            {
                traceback[i][j] = 3;
            }
        }
    }

    TimePoint end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "SW Kernel executed in " << elapsed.count() << " seconds.\n";

    // Traceback to get the aligned sequences
    std::string alignedSeqA, alignedSeqB;
    int i = maxRow, j = maxCol;
    while (i > 0 && j > 0 && scoreMatrix[i][j] != 0)
    {
        if (traceback[i][j] == 1)
        {
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            i--;
            j--;
        }
        else if (traceback[i][j] == 2)
        {
            alignedSeqA = seqA[i - 1] + alignedSeqA;
            alignedSeqB = '-' + alignedSeqB;
            i--;
        }
        else if (traceback[i][j] == 3)
        {
            alignedSeqA = '-' + alignedSeqA;
            alignedSeqB = seqB[j - 1] + alignedSeqB;
            j--;
        }
    }

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
