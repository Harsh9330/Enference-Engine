# Phase 2: Binary Weight Format Specification

The `weights.bin` file is a flat binary blob. There is no structural metadata interleaved with the weights. The layout is derived externally via `manifest.json`.

## File Structure
1. **Header (8 Bytes):**
   - Bytes 0-3: Magic Number `HVRP` (ASCII chars).
   - Bytes 4-7: Version Number `1` (32-bit little-endian unsigned integer).
2. **Payload (Variable Length):**
   - A sequence of raw tensor byte arrays.
   - **Crucial Rule:** Every tensor begins at an offset perfectly aligned to a **32-byte boundary** from the start of the file.
   - Any gap between the end of one tensor and the 32-byte aligned start of the next tensor is padded with `0x00`.

## Memory Layout
- All arrays are explicitly cast to `float32`.
- Matrices (2D arrays) are stored in **Row-Major (C-contiguous)** format.
- Dimension convention is pinned as `[in_features, out_features]`.
- C++ GEMM operations should directly compute `y = xW` without any runtime transposition.

## C++ Reader Struct Equivalent
```cpp
// Check header for 'HVRP' and version 1 before proceeding.
struct TensorIndex {
    std::string name;
    std::vector<int> shape;
    size_t offset;  // Guaranteed to be (offset % 32 == 0)
    size_t length;
};
// Use mmap() to load the entire file into a read-only buffer, 
// then cast pointers to (float*) at the offsets specified by manifest.json.