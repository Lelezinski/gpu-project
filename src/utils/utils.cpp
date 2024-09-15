#include "utils.hpp"

// Function to parse the config file
bool parseConfig(const std::string &filepath, std::string &databasePath, int &match, int &mismatch, int &gap_penalty)
{
    std::ifstream file(filepath);
    
    if (!file.is_open())
    {
        std::cerr << "Error opening config file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t equalPos = line.find('=');
        if (equalPos != std::string::npos)
        {
            std::string key = line.substr(0, equalPos);
            std::string value = line.substr(equalPos + 1);

            key.erase(key.find_last_not_of(" \n\r\t") + 1);
            value.erase(0, value.find_first_not_of(" \n\r\t"));

            if (key == "match")
            {
                match = std::stoi(value);
            }
            else if (key == "mismatch")
            {
                mismatch = std::stoi(value);
            }
            else if (key == "gap")
            {
                gap_penalty = std::stoi(value);
            }
            else if (key == "database")
            {
                databasePath = value;
            }
        }
    }
    
    file.close();

    return true;
}

// Function to read sequences from the database file, removing newline characters
std::vector<std::string> readDatabase(const std::string &filepath)
{
    std::ifstream file(filepath);
    std::vector<std::string> database;

    if (!file.is_open())
    {
        std::cerr << "Error opening database file: " << filepath << std::endl;
        return database;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Remove any newline characters ('\n' or '\r') from the line
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        if (!line.empty())
        {
            database.push_back(line);
        }
    }

    file.close();
    return database;
}

// Function to print the result of the alignment
void printResult(const SWAResult &result)
{
    std::cout << "Aligned Sequences:\n";

    std::string alignedSeqA = result.alignedSeqA;
    std::string alignedSeqB = result.alignedSeqB;

    // Print the sequences with color-coding
    for (size_t i = 0; i < alignedSeqA.length(); ++i)
    {
        char baseA = alignedSeqA[i];
        char baseB = alignedSeqB[i];

        if (baseA == '-' || baseB == '-')
        {
            std::cout << YELLOW << baseA << RESET;
        }
        else if (baseA == baseB)
        {
            std::cout << GREEN << baseA << RESET;
        }
        else
        {
            std::cout << RED << baseA << RESET;
        }
    }
    std::cout << "\n";

    std::cout << "Alignment Score: " << result.score << "\n";
}

// Function to check if a given sequence is a valid DNA sequence
bool isValidDNASequence(const std::string &sequence)
{
    for (char nucleotide : sequence)
    {
        // Check if the character is not one of the valid DNA bases (A, C, G, T)
        if (nucleotide != 'A' && nucleotide != 'C' && nucleotide != 'G' && nucleotide != 'T')
        {
            std::cerr << "Invalid character in sequence: " << nucleotide << std::endl;
            return false;
        }
    }
    return true;
}