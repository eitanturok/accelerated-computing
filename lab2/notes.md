# Commands

To run the cuda file, do
```bash
nvcc fma_latency.cu -o fma_latency && ./fma_latency
```

To run the cuda file and see its SASS, do
```
// generate SASS during compilation
nvcc fma_latency.cu -o fma_latency -lineinfo

// inspect the SASS
cuobjdump --dump-sass fma_latency > sass_output.txt

// run the program
./fma_latency
```
or all in one line
```bash
nvcc fma_latency.cu -o fma_latency -lineinfo && cuobjdump --dump-sass fma_latency > sass_output.txt && ./fma_latency
```

# Notes

What exactly is FFMA? Is it PTX, SASS? How do we write it in CUDA if the instruction exists at a lower level?

1. FMA = fused multiply add. this is both a general concept and the name of the instruction on the CPU. FFMA = floating-point fused multiply add is the name of the FMA instruction on the GPU.
2. FFMA itself is a SASS instruction, not a PTX instruction or a command in cuda.
3. To actually use an FFMA instruction, use `fmaf()` cmd in cuda ([docs](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff)).
4. Behind the scenes `fmaf` CUDA gets turned into FFMA SASS instruction with the pipeline
    a. CUDA C++: `fmaf()`
    b. PTX (virtual ISA): `fma.rn.f32`
    c. SASS (real GPU assembly): `FFMA`
5. To

We have two tools to inspect compiled cuda binaries ([docs](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)):
1. `cuobjump` - accepts both cubin files and host binaries but has limited output
2. `nvdisasm` - accepts ony cubin files but provides richer output


The flag `-arch=native` means nvcc auto-detects the GPU.

Look at SASS, we have:
1. `LDC`: load from constant memory
2. `LDCU`: load from constant memory into a uniform register. A uniform register is shared across all threads in a warp.
3. `LDG`: load from global memory. This is my `d*` dereference.
4. `CS2R`: copy special register to register. This is when we read the clock cycle into a register.
5. `STG`: store to global memory
6. `MEMBAR` / `ERRBAR` / `CGAERRBAR`: memory barrier, error barrier instructions from `__threadfence()`
7. `NOP`: bloackwell requires instruction bundles to be aligned to certain boundaries so the assembler fills unused space with `NOP`s.

We are given the hint
> Helpful Tip: To make the timing more stable, chain dependent FMA instructions so the compiler cannot reorder them. Finally, divide the total execution time by the number of operations to compute the average latency per instruction.
This means that we should do a FMA several times and make each FMA's output feed into the next one's input so that the FMA calls are all dependent on each other and cannot be reordered. Then we'll divide the total cycles by the number of FMAs.
