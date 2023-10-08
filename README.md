# minictc

A minimalistic CTC implementation in C++ for CPUs. It's tested to yield numerically equivalent results against PyTorch 2.0 on some trivial examples, but seems to be several times quicker. This is mostly for educational purposes and intends to keep the code somewhat simple and close to CTC equations in [Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

To build it, use CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest
```

Check the test source code for actual usage examples.