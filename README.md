# Discretized Lloyd's algorithm
This project presents an algorithm for computing Voronoi Diagrams of a mesh's surface using a discrete clustering approach.

The algorithm is mainly inspierd by Lloyd's, and only works by simple graph operations on a graph structure built from the input mesh.

## Compile
### Linux / MacOS
Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

### Windows / Visual Studio 2022
Type the following commands in the prompt:

    mkdir build
    cd build
    cmake .. -G "Visual Studio 17 2022"
