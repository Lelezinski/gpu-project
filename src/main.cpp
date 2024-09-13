#include "inc/swa.hpp"
#include "inc/utils.hpp"
#include <iostream>
#include <vector>

#define CONFIG_FILE "./config.cfg"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <query_sequence>\n";
        return 1;
    }

    std::string querySeq = argv[1];
    std::string configFilePath = CONFIG_FILE;
    std::string databaseFilePath;
    int match, mismatch, gap;

    // Parse the config file
    if (!parseConfig(configFilePath, databaseFilePath, match, mismatch, gap))
    {
        std::cerr << "Failed to parse configuration file.\n";
        return 1;
    }

    // Read the database sequences
    std::vector<std::string> database = readDatabase(databaseFilePath);
    if (database.empty())
    {
        std::cerr << "No sequences found in the database file.\n";
        return 1;
    }

    // Validate query sequence
    if (!isValidDNASequence(querySeq))
    {
        std::cerr << "Query sequence is invalid." << std::endl;
        return 1;
    }

    // Validate database sequences
    for (const std::string &dbSeq : database)
    {
        if (!isValidDNASequence(dbSeq))
        {
            std::cerr << "Database contains invalid sequences." << std::endl;
            return 1;
        }
    }

    // Variables to store the best result
    SWAResult bestResult;
    int bestScore = -1;
    int bestDbIndex = -1;

    // Compare query with each sequence in the database
    for (int i = 0; i < database.size(); ++i)
    {
        SWAResult result = smithWaterman(querySeq, database[i], match, mismatch, gap);

        // Track the best alignment score
        if (result.score > bestScore)
        {
            bestScore = result.score;
            bestResult = result;
            bestDbIndex = i;
        }
    }

    // Print the best result
    if (bestScore != -1)
    {
        std::cout << "Best alignment found with sequence #" << bestDbIndex + 1 << " in the database:\n";
        printResult(bestResult);
    }
    else
    {
        std::cerr << "No valid alignments found.\n";
    }

    return 0;
}
