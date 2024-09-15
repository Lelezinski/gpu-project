#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "swa.hpp"

// ANSI color codes
const std::string GREEN = "\033[32m";
const std::string RED = "\033[31m";
const std::string YELLOW = "\033[33m";
const std::string RESET = "\033[0m";

bool parseConfig(const std::string &filepath, std::string &databasePath, int &match, int &mismatch, int &gap_penalty);
void printResult(const SWAResult &result);
std::vector<std::string> readDatabase(const std::string &filepath);
bool isValidDNASequence(const std::string &sequence);
