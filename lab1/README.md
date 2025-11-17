Starter code for 6.S894 [Lab 1](https://accelerated-computing-class.github.io/fall24/labs/lab1).

Resources:
1. https://gist.github.com/MangaD/1fad63756ad8c946ce01dd1d52eff173

Notes:
1. SIMD = single instruction, multiple data. This means instead of processing one data value at a time for every instruction, we can process multiple dat values for every instruction.
2. How do use SIMD?
    a. *Compiler*
        i. Let the compiler take care of it.
        ii. Add flags like `-O2 -march=native` or `-O3` to the gcc compiler or `/O2 /arch:AVX2` to the MSVC compiler to automatically enable vectorization.
        iii. You can even use compiler hints like `#pragma omp simd` to force the compiler to use SIMD execution even when the compiler is unsure if it should do so.
        iv. Pro: easy to implement. Con: no fine-grained control, not best optimizations.
    b. *Intrinsics*
        i. Manually add SIMD intrinsics
        ii. Use the AVX2 header `#include <immintrin.h>` to specify exactly how, where we vectorize.
        iii. Note: this is not portable, i.e. it only works on machines with AVX2 support.
        iv. Pro: fine-grained control, can do lots of optimizations. Cons: tons of manual effort.
    c. *SIMD libraries:*
        i. Use libraries like `valarray`, `Eigen`, or `xsimd`
        ii. These libraries offer a middle-ground. They give you some abstractions over SIMD intrinsics so it is not as low level and make things portable. At the same time, they probably cannot get the same speed ups as intrinsics because they don't have the same level of fine-grained control.

AVX2-512 intrinsic types:

| Type | Size | Purpose | Capacity |
|------|------|---------|----------|
| `__m512i` | 512-bit | Packed integers (unspecified signedness) | 64×8-bit, 32×16-bit, 16×32-bit, or 8×64-bit |
| `__m512i_u` | 512-bit | Unaligned packed integers | 64×8-bit, 32×16-bit, 16×32-bit, or 8×64-bit |
| `__m512` | 512-bit | Packed single-precision floats (32-bit) | 16×32-bit floats |
| `__m512_u` | 512-bit | Unaligned packed single-precision floats | 16×32-bit floats |
| `__m512d` | 512-bit | Packed double-precision floats (64-bit) | 8×64-bit doubles |
| `__m512d_u` | 512-bit | Unaligned packed double-precision floats | 8×64-bit doubles |
| `__mmask8` | 8-bit | Mask register for conditional operations | 8 lanes |
| `__mmask16` | 16-bit | Mask register for conditional operations | 16 lanes |
| `__mmask32` | 32-bit | Mask register for conditional operations | 32 lanes |
| `__mmask64` | 64-bit | Mask register for conditional operations | 64 lanes |

Q: how do you specify if an integer is signed or unsigned with these types?
A: All the AVX2-512 intrinsic types are signedness-agnostic — it doesn't inherently specify signed or unsigned. The signedness is determined by the intrinsic function you use, not the type itself.

```cpp
// data of type __m512i does not have an explicit sign
__m512i data = _mm512_loadu_si512(ptr);

// Treat as signed integers
__m512i result1 = _mm512_add_epi32(data, data);  // signed addition

// Treat as unsigned integers
__m512i result2 = _mm512_add_epu32(data, data);  // unsigned addition
```
