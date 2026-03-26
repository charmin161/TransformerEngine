#include <cuda_runtime.h>
#include <stdint.h>

struct alignas(16) Tcgen05Nvfp4InstructionOperands
{
	uint32_t destinationTensorMemoryColumn;
	uint32_t scaleATensorMemoryColumn;
	uint32_t scaleBTensorMemoryColumn;
	uint32_t enableInputD;
	uint64_t operandADescriptor;
	uint64_t operandBDescriptor;
	uint64_t instructionDescriptor;
};

__device__ __forceinline__ void ExecuteTcgen05Nvfp4Mma(
	const Tcgen05Nvfp4InstructionOperands& instructionOperands)
{
	asm volatile(
		"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X \n\t"
		"  [%0], %1, %2, %3, [%4], [%5], %6;\n"
		:
		: "r"(instructionOperands.destinationTensorMemoryColumn),
		  "l"(instructionOperands.operandADescriptor),
		  "l"(instructionOperands.operandBDescriptor),
		  "l"(instructionOperands.instructionDescriptor),
		  "r"(instructionOperands.scaleATensorMemoryColumn),
		  "r"(instructionOperands.scaleBTensorMemoryColumn),
		  "r"(instructionOperands.enableInputD)
		: "memory");
}

extern "C" __global__ void Tcgen05Nvfp4MmaKernel(
	const Tcgen05Nvfp4InstructionOperands* instructionStream)
{
	if (blockIdx.x != 0)
	{
		return;
	}

	const Tcgen05Nvfp4InstructionOperands instructionOperands = instructionStream[0];
	ExecuteTcgen05Nvfp4Mma(instructionOperands);
}
