#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "swa.hpp"

bool parseConfig(const std::string &filepath, std::string &databasePath, int &match, int &mismatch, int &gap_penalty);
void printResult(const SWAResult &result);
std::vector<std::string> readDatabase(const std::string &filepath);
bool isValidDNASequence(const std::string &sequence);
