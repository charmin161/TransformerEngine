#include <cuda_runtime.h>
#include <stdint.h>

struct alignas(16) Nvfp4WarpTileOperands
{
	uint32_t operandARegisters[4];
	uint32_t operandBRegisters[2];
	uint32_t scaleAData;
	uint32_t scaleBData;
	uint32_t scaleABlockId;
	uint32_t scaleAThreadId;
	uint32_t scaleBBlockId;
	uint32_t scaleBThreadId;
};

struct alignas(16) Nvfp4WarpTileAccumulator
{
	float accumulatorRegisters[4];
};

__device__ __forceinline__ Nvfp4WarpTileAccumulator ExecuteNvfp4WarpLevelMma(
	const Nvfp4WarpTileOperands& packedOperands,
	const Nvfp4WarpTileAccumulator& inputAccumulator)
{
	Nvfp4WarpTileAccumulator outputAccumulator{};

	asm volatile(
		"{\n\t"
		" .reg .u16 scaleABlockIdU16;\n\t"
		" .reg .u16 scaleAThreadIdU16;\n\t"
		" .reg .u16 scaleBBlockIdU16;\n\t"
		" .reg .u16 scaleBThreadIdU16;\n\t"
		" cvt.u16.u32 scaleABlockIdU16, %14;\n\t"
		" cvt.u16.u32 scaleAThreadIdU16, %15;\n\t"
		" cvt.u16.u32 scaleBBlockIdU16, %16;\n\t"
		" cvt.u16.u32 scaleBThreadIdU16, %17;\n\t"
		" mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \n\t"
		"   {%0, %1, %2, %3}, \n\t"
		"   {%4, %5, %6, %7}, \n\t"
		"   {%8, %9}, \n\t"
		"   {%10, %11, %12, %13}, \n\t"
		"   %18, {scaleABlockIdU16, scaleAThreadIdU16}, %19, {scaleBBlockIdU16, scaleBThreadIdU16};\n\t"
		"}\n"
		: "=f"(outputAccumulator.accumulatorRegisters[0]),
		  "=f"(outputAccumulator.accumulatorRegisters[1]),
		  "=f"(outputAccumulator.accumulatorRegisters[2]),
		  "=f"(outputAccumulator.accumulatorRegisters[3])
		: "r"(packedOperands.operandARegisters[0]),
		  "r"(packedOperands.operandARegisters[1]),
		  "r"(packedOperands.operandARegisters[2]),
		  "r"(packedOperands.operandARegisters[3]),
		  "r"(packedOperands.operandBRegisters[0]),
		  "r"(packedOperands.operandBRegisters[1]),
		  "f"(inputAccumulator.accumulatorRegisters[0]),
		  "f"(inputAccumulator.accumulatorRegisters[1]),
		  "f"(inputAccumulator.accumulatorRegisters[2]),
		  "f"(inputAccumulator.accumulatorRegisters[3]),
		  "r"(packedOperands.scaleABlockId),
		  "r"(packedOperands.scaleAThreadId),
		  "r"(packedOperands.scaleBBlockId),
		  "r"(packedOperands.scaleBThreadId),
		  "r"(packedOperands.scaleAData),
		  "r"(packedOperands.scaleBData));

	return outputAccumulator;
}

extern "C" __global__ void Nvfp4WarpLevelMmaKernel(
	const Nvfp4WarpTileOperands* inputTiles,
	const Nvfp4WarpTileAccumulator* inputAccumulators,
	Nvfp4WarpTileAccumulator* outputAccumulators)
{
	if (blockIdx.x != 0 || threadIdx.x >= warpSize)
	{
		return;
	}

	const Nvfp4WarpTileOperands packedOperands = inputTiles[0];
	const Nvfp4WarpTileAccumulator inputAccumulator = inputAccumulators[0];
	const Nvfp4WarpTileAccumulator outputAccumulator = ExecuteNvfp4WarpLevelMma(packedOperands, inputAccumulator);
	outputAccumulators[threadIdx.x] = outputAccumulator;
}
