#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK_CUDA(call) \
	do \
	{ \
		cudaError_t cudaError = (call); \
		if (cudaError != cudaSuccess) \
		{ \
			std::fprintf(stderr, "CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaError)); \
			return 1; \
		} \
	} while (0)

__global__ void KeepGpuUtilKernel(volatile unsigned long long* deviceSink, unsigned int sleepCycles)
{
	unsigned long long state = clock64() ^ threadIdx.x ^ blockIdx.x;

	while (true)
	{
#pragma unroll 128
		for (int iteration = 0; iteration < 128; ++iteration)
		{
			state = state * 2862933555777941757ULL + 3037000493ULL;
		}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
		if (sleepCycles > 0)
		{
			__nanosleep(sleepCycles);
		}
#endif

		if ((state & 0xfffffULL) == 0)
		{
			deviceSink[0] = state;
		}
	}
}

int main(int argumentCount, char** arguments)
{
	int deviceId = argumentCount > 1 ? std::atoi(arguments[1]) : 0;
	int blockCount = argumentCount > 2 ? std::atoi(arguments[2]) : 1;
	unsigned int sleepCycles = argumentCount > 3 ? static_cast<unsigned int>(std::atoi(arguments[3])) : 1000;

	CHECK_CUDA(cudaSetDevice(deviceId));

	int leastPriority = 0;
	int greatestPriority = 0;
	CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

	cudaStream_t lowPriorityStream = nullptr;
	CHECK_CUDA(cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority));

	unsigned long long* deviceSink = nullptr;
	CHECK_CUDA(cudaMalloc(&deviceSink, sizeof(unsigned long long)));
	CHECK_CUDA(cudaMemsetAsync(deviceSink, 0, sizeof(unsigned long long), lowPriorityStream));

	std::printf("Device: %d\n", deviceId);
	std::printf("Blocks: %d\n", blockCount);
	std::printf("Threads per block: 1\n");
	std::printf("Sleep cycles: %u\n", sleepCycles);
	std::printf("Press Ctrl+C to stop.\n");

	KeepGpuUtilKernel<<<blockCount, 1, 0, lowPriorityStream>>>(deviceSink, sleepCycles);
	CHECK_CUDA(cudaPeekAtLastError());

	while (true)
	{
		cudaError_t streamStatus = cudaStreamQuery(lowPriorityStream);

		if (streamStatus == cudaSuccess)
		{
			break;
		}

		if (streamStatus != cudaErrorNotReady)
		{
			std::fprintf(stderr, "CUDA stream error: %s\n", cudaGetErrorString(streamStatus));
			return 1;
		}

		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	return 0;
}
