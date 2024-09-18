# GPU-Accelerated Bioinformatics: The Smith-Waterman Algorithm

This project implements the Smith-Waterman algorithm for local sequence alignment, using both CPU and GPU (CUDA) versions to enhance performance. The GPU versions take advantage of parallel processing and memory optimizations, specifically tuned for devices like the Jetson Nano.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running the Program](#running-the-program)

## Overview
The Smith-Waterman algorithm performs local sequence alignment by comparing segments of sequences to identify regions of similarity. This implementation includes:
- A CPU-only version.
- A GPU version using CUDA to exploit parallelism.
- An optimized GPU version using shared memory and tailored block/thread configurations for maximum performance on the Jetson Nano.

## Prerequisites
Before running the program, ensure that you have:
- CUDA Toolkit installed (for GPU versions).
- A CUDA-capable GPU (the program is optimized for the Jetson Nano).
- `make` utility for building the project.
- C++17 compatible compiler.

## Setup
Use the `make` command to compile the different versions of the program. This will generate multiple executables corresponding to different implementations:
- `smith_waterman_cpu`: CPU-only version.
- `smith_waterman_gpu`: Basic GPU version.
- `smith_waterman_gpuopt`: Optimized GPU version with shared memory and fine-tuned thread-block configuration.

```bash
$ make
```

Before running the program, you need to configure the config.cfg file. This file contains parameters for the alignment algorithm, such as the scoring scheme (match, mismatch, gap penalty) and the path to the sequence database.
Database Format

The database should contain one or more DNA sequences, with the following requirements:

- Each sequence must be composed of the characters A, C, G, and T.
- All sequences (both the query and database sequences) must be of the same length.

## Running the Program
To run the program, use the corresponding executable for the version you want to test:

```bash
$ smith_waterman_{cpu|gpu|gpuopt1} {QUERY}
```

Where:
- `cpu`, `gpu`, or `gpuopt1` specifies the version of the program you wish to run.
- `{QUERY}` is the sequence string to be aligned against the sequences in the database (e.g., ACGT).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Lorenzo Ruotolo, s313207