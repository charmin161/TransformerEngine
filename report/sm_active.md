#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <vector>
#include <chrono>
#include <csignal>
#include <algorithm>

#define CUDA_CHECK(cudaCall) \
	do \
	{ \
		cudaError_t cudaError = (cudaCall); \
		if (cudaError != cudaSuccess) \
		{ \
			std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaError)); \
			std::exit(EXIT_FAILURE); \
		} \
	} while (0)

volatile std::sig_atomic_t shouldStop = 0;

void HandleSignal(int signalNumber)
{
	shouldStop = 1;
}

__global__ void KeepGpuUtilActiveSleepKernel(std::uint64_t* outputValue, unsigned long long durationCycles, unsigned int sleepNanoseconds)
{
	unsigned long long startClock = clock64();
	unsigned long long currentClock = startClock;

	std::uint64_t accumulator =
		0x9E3779B97F4A7C15ULL ^
		static_cast<std::uint64_t>(blockIdx.x) ^
		static_cast<std::uint64_t>(threadIdx.x);

	while ((currentClock - startClock) < durationCycles)
	{
#if __CUDA_ARCH__ >= 700
		__nanosleep(sleepNanoseconds);
#else
		#pragma unroll 4
		for (int iterationIndex = 0; iterationIndex < 16; ++iterationIndex)
		{
			accumulator ^= accumulator << 13;
			accumulator ^= accumulator >> 7;
			accumulator ^= accumulator << 17;
			asm volatile("" : "+l"(accumulator));
		}
#endif
		currentClock = clock64();
	}

	if (threadIdx.x == 0)
	{
		outputValue[0] = accumulator ^ currentClock;
	}
}

void RunOnDevice(
	int deviceIndex,
	int totalSeconds,
	int kernelMilliseconds,
	int threadsPerBlock,
	unsigned int sleepNanoseconds)
{
	CUDA_CHECK(cudaSetDevice(deviceIndex));

	int clockRateKhz = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKhz, cudaDevAttrClockRate, deviceIndex));

	cudaDeviceProp deviceProperties;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));

	const int blockCount = 1;
	const std::size_t resultBytes = sizeof(std::uint64_t);

	const unsigned long long clockRateHz =
		static_cast<unsigned long long>(clockRateKhz) * 1000ULL;

	const unsigned long long durationCycles =
		(clockRateHz / 1000ULL) * static_cast<unsigned long long>(kernelMilliseconds);

	std::uint64_t* deviceOutputValue = nullptr;
	CUDA_CHECK(cudaMalloc(&deviceOutputValue, resultBytes));

	std::printf(
		"[GPU %d] Device: %s, blocks: %d, threads: %d, allocation: %.2f KB, kernel slice: %d ms, nanosleep: %u ns\n",
		deviceIndex,
		deviceProperties.name,
		blockCount,
		threadsPerBlock,
		static_cast<double>(resultBytes) / 1024.0,
		kernelMilliseconds,
		sleepNanoseconds);

	const auto endTime = std::chrono::steady_clock::now() + std::chrono::seconds(totalSeconds);

	while (!shouldStop && std::chrono::steady_clock::now() < endTime)
	{
		KeepGpuUtilActiveSleepKernel<<<blockCount, threadsPerBlock>>>(deviceOutputValue, durationCycles, sleepNanoseconds);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	CUDA_CHECK(cudaFree(deviceOutputValue));
	CUDA_CHECK(cudaDeviceReset());

	std::printf("[GPU %d] Finished\n", deviceIndex);
}

int main(int argumentCount, char** argumentValues)
{
	std::signal(SIGINT, HandleSignal);
	std::signal(SIGTERM, HandleSignal);

	int totalSeconds = 360000;
	int kernelMilliseconds = 1000;
	int threadsPerBlock = 1;
	unsigned int sleepNanoseconds = 1000000;

	if (argumentCount >= 2)
	{
		totalSeconds = std::atoi(argumentValues[1]);
	}

	if (argumentCount >= 3)
	{
		kernelMilliseconds = std::atoi(argumentValues[2]);
	}

	if (argumentCount >= 4)
	{
		threadsPerBlock = std::atoi(argumentValues[3]);
	}

	if (argumentCount >= 5)
	{
		sleepNanoseconds = static_cast<unsigned int>(std::atoi(argumentValues[4]));
	}

	totalSeconds = std::max(1, totalSeconds);
	kernelMilliseconds = std::max(1, kernelMilliseconds);
	threadsPerBlock = std::max(1, threadsPerBlock);
	sleepNanoseconds = std::max(1u, sleepNanoseconds);

	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0)
	{
		std::fprintf(stderr, "No CUDA devices found.\n");
		return EXIT_FAILURE;
	}

	std::printf("Visible CUDA device count: %d\n", deviceCount);
	std::printf("Total duration: %d seconds\n", totalSeconds);
	std::printf("Kernel slice duration: %d ms\n", kernelMilliseconds);
	std::printf("Threads per block: %d\n", threadsPerBlock);
	std::printf("Nanosleep per loop: %u ns\n", sleepNanoseconds);
	std::printf("Mode: minimal GPU-Util filler with nanosleep\n");

	std::vector<std::thread> workerThreads;

	for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
	{
		workerThreads.emplace_back(
			RunOnDevice,
			deviceIndex,
			totalSeconds,
			kernelMilliseconds,
			threadsPerBlock,
			sleepNanoseconds);
	}

	for (std::thread& workerThread : workerThreads)
	{
		if (workerThread.joinable())
		{
			workerThread.join();
		}
	}

	return EXIT_SUCCESS;
}
