# NVFP4 PTX / SASS Inspection Demo

This directory contains two minimal CUDA source files intended for PTX and SASS inspection:

- `nvfp4_mma_sync_minimal.cu`
	- Carries the warp-level instruction:
	- `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
- `tcgen05_mma_nvf4_minimal.cu`
	- Carries the Blackwell TCGen05 instruction:
	- `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`

## Important caveat

The first file is a fragment-level instruction carrier kernel.
The second file is a descriptor-level instruction carrier kernel.

Both are intentionally minimal so the generated PTX and SASS are easy to inspect.
They are not meant to be numerically correct end-to-end GEMM samples without adding real fragment packing, tensor-memory allocation, valid descriptors, and valid scale-factor layouts.

## Generate PTX and SASS

Run:

```bash
./build_ptx_and_sass.sh
```

Expected outputs in `generated/`:

- `nvfp4_mma_sync_minimal.ptx`
- `nvfp4_mma_sync_minimal.cubin`
- `nvfp4_mma_sync_minimal.nvdisasm.sass`
- `nvfp4_mma_sync_minimal.cuobjdump.sass`
- `tcgen05_mma_nvf4_minimal.ptx`
- `tcgen05_mma_nvf4_minimal.cubin`
- `tcgen05_mma_nvf4_minimal.nvdisasm.sass`
- `tcgen05_mma_nvf4_minimal.cuobjdump.sass`

## Suggested grep targets

```bash
grep -n "mma.sync.aligned" generated/nvfp4_mma_sync_minimal.ptx
grep -n "tcgen05.mma" generated/tcgen05_mma_nvf4_minimal.ptx
grep -n "MMA" generated/nvfp4_mma_sync_minimal.nvdisasm.sass
grep -n "TCGEN05" generated/tcgen05_mma_nvf4_minimal.nvdisasm.sass
```

## Toolkit expectations

Use a recent CUDA Toolkit that supports Blackwell targets such as `sm_100a` and `sm_120a`, plus `nvdisasm` and `cuobjdump` from the CUDA Binary Utilities package.
