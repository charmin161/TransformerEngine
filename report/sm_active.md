#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <vector>
#include <chrono>
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

__global__ void LimitedSmActiveKernel(std::uint64_t* blockResults, unsigned long long durationCycles)
{
	const unsigned int threadIndex = threadIdx.x;
	const unsigned int blockIndex = blockIdx.x;

	unsigned long long startClock = clock64();
	unsigned long long currentClock = startClock;

	std::uint64_t accumulator =
		0x9E3779B97F4A7C15ULL ^
		(static_cast<std::uint64_t>(blockIndex) << 32) ^
		static_cast<std::uint64_t>(threadIndex);

	while ((currentClock - startClock) < durationCycles)
	{
		#pragma unroll 64
		for (int iterationIndex = 0; iterationIndex < 256; ++iterationIndex)
		{
			accumulator ^= accumulator << 13;
			accumulator ^= accumulator >> 7;
			accumulator ^= accumulator << 17;
			asm volatile("" : "+l"(accumulator));
		}

		currentClock = clock64();
	}

	if (threadIndex == 0)
	{
		blockResults[blockIndex] = accumulator;
	}
}

void RunOnDevice(
	int deviceIndex,
	int totalSeconds,
	int targetSmPercent,
	int busyMilliseconds,
	int periodMilliseconds)
{
	CUDA_CHECK(cudaSetDevice(deviceIndex));

	int smCount = 0;
	int clockRateKhz = 0;

	CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceIndex));
	CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKhz, cudaDevAttrClockRate, deviceIndex));

	cudaDeviceProp deviceProperties;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));

	const int threadsPerBlock = 32;
	const int blockCount = std::max(1, (smCount * targetSmPercent + 99) / 100);

	const unsigned long long clockRateHz =
		static_cast<unsigned long long>(clockRateKhz) * 1000ULL;

	const unsigned long long busyCycles =
		(clockRateHz / 1000ULL) * static_cast<unsigned long long>(busyMilliseconds);

	std::uint64_t* deviceBlockResults = nullptr;
	const std::size_t resultBytes = static_cast<std::size_t>(blockCount) * sizeof(std::uint64_t);

	CUDA_CHECK(cudaMalloc(&deviceBlockResults, resultBytes));

	std::printf(
		"[GPU %d] Device: %s, SMs: %d, blocks: %d, allocation: %.2f KB\n",
		deviceIndex,
		deviceProperties.name,
		smCount,
		blockCount,
		static_cast<double>(resultBytes) / 1024.0);

	const auto endTime = std::chrono::steady_clock::now() + std::chrono::seconds(totalSeconds);

	while (std::chrono::steady_clock::now() < endTime)
	{
		LimitedSmActiveKernel<<<blockCount, threadsPerBlock>>>(deviceBlockResults, busyCycles);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		const int sleepMilliseconds = periodMilliseconds - busyMilliseconds;

		if (sleepMilliseconds > 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepMilliseconds));
		}
	}

	CUDA_CHECK(cudaFree(deviceBlockResults));

	std::printf("[GPU %d] Finished\n", deviceIndex);
}

int main(int argumentCount, char** argumentValues)
{
	int totalSeconds = 60;
	int targetSmPercent = 20;
	int busyMilliseconds = 50;
	int periodMilliseconds = 100;

	if (argumentCount >= 2)
	{
		totalSeconds = std::atoi(argumentValues[1]);
	}

	if (argumentCount >= 3)
	{
		targetSmPercent = std::atoi(argumentValues[2]);
	}

	if (argumentCount >= 4)
	{
		busyMilliseconds = std::atoi(argumentValues[3]);
	}

	if (argumentCount >= 5)
	{
		periodMilliseconds = std::atoi(argumentValues[4]);
	}

	targetSmPercent = std::max(1, std::min(targetSmPercent, 100));
	busyMilliseconds = std::max(1, busyMilliseconds);
	periodMilliseconds = std::max(busyMilliseconds, periodMilliseconds);

	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0)
	{
		std::fprintf(stderr, "No CUDA devices found.\n");
		return EXIT_FAILURE;
	}

	std::printf("Visible CUDA device count: %d\n", deviceCount);
	std::printf("Total duration: %d seconds\n", totalSeconds);
	std::printf("Target SM percent per GPU: %d%%\n", targetSmPercent);
	std::printf("Busy milliseconds: %d\n", busyMilliseconds);
	std::printf("Period milliseconds: %d\n", periodMilliseconds);
	std::printf("Estimated average extra SM active per GPU: %.2f%%\n",
		static_cast<double>(targetSmPercent) *
		static_cast<double>(busyMilliseconds) /
		static_cast<double>(periodMilliseconds));

	std::vector<std::thread> workerThreads;

	for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
	{
		workerThreads.emplace_back(
			RunOnDevice,
			deviceIndex,
			totalSeconds,
			targetSmPercent,
			busyMilliseconds,
			periodMilliseconds);
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
