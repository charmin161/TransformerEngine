[rank0]:[E512 10:21:08.164595040 ProcessGroupNCCL.cpp:2057] [PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 0] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xede05571c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xede0557d3574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xede05636e0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xede05638d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xede056394460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xede056395e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xede088291ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xede0894d595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xede08953bb0c in /lib/aarch64-linux-gnu/libc.so.6)

[rank1]:[E512 10:21:08.164594976 ProcessGroupNCCL.cpp:2057] [PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xe5174fc0c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xe5174fcc3574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xe5175085e0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xe5175087d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xe51750884460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xe51750885e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xe51782781ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xe517839c595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xe51783a2bb0c in /lib/aarch64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
terminate called after throwing an instance of 'c10::DistBackendError'
[rank3]:[E512 10:21:08.164684993 ProcessGroupNCCL.cpp:2057] [PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 3] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xec06adb9c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xec06adc53574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xec06ae7ee0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xec06ae80d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xec06ae814460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xec06ae815e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xec06e0711ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xec06e195595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xec06e19bbb0c in /lib/aarch64-linux-gnu/libc.so.6)

[rank2]:[E512 10:21:08.164689473 ProcessGroupNCCL.cpp:2057] [PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 2] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xeba71106c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xeba711123574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xeba711cbe0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xeba711cdd404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xeba711ce4460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xeba711ce5e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xeba743be1ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xeba744e2595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xeba744e8bb0c in /lib/aarch64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 0] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xede05571c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xede0557d3574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xede05636e0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xede05638d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xede056394460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xede056395e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xede088291ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xede0894d595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xede08953bb0c in /lib/aarch64-linux-gnu/libc.so.6)

Exception raised from run at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2063 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xede05571c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xebf100 (0xede05634f100 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::Watchdog::run() + 0x474 (0xede0563961c4 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xe1ae0 (0xede088291ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #4: <unknown function> + 0x8595c (0xede0894d595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #5: <unknown function> + 0xebb0c (0xede08953bb0c in /lib/aarch64-linux-gnu/libc.so.6)
  what():  
[PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xe5174fc0c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xe5174fcc3574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xe5175085e0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xe5175087d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xe51750884460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xe51750885e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xe51782781ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xe517839c595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xe51783a2bb0c in /lib/aarch64-linux-gnu/libc.so.6)

Exception raised from run at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2063 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xe5174fc0c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xebf100 (0xe5175083f100 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::Watchdog::run() + 0x474 (0xe517508861c4 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xe1ae0 (0xe51782781ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #4: <unknown function> + 0x8595c (0xe517839c595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #5: <unknown function> + 0xebb0c (0xe51783a2bb0c in /lib/aarch64-linux-gnu/libc.so.6)
Fatal Python error: 
Aborted

Thread 0xFatal Python error: 0000eddb21fbf180Aborted (most recent call first):


  <no Python frame>
Thread 0x
Thread 0x0000e511dd72f1800000eddb227cf180 (most recent call first):
 (most recent call first):
  <no Python frame>
  <no Python frame>


  what():  Thread 0xThread 0x[PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 3] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xec06adb9c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xec06adc53574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xec06ae7ee0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xec06ae80d404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xec06ae814460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xec06ae815e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xec06e0711ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xec06e195595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xec06e19bbb0c in /lib/aarch64-linux-gnu/libc.so.6)

Exception raised from run at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2063 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xec06adb9c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xebf100 (0xec06ae7cf100 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::Watchdog::run() + 0x474 (0xec06ae8161c4 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xe1ae0 (0xec06e0711ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #4: <unknown function> + 0x8595c (0xec06e195595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #5: <unknown function> + 0xebb0c (0xec06e19bbb0c in /lib/aarch64-linux-gnu/libc.so.6)
0000e511ddf3f180  what():  0000eddb22fdf180
 (most recent call first):
[PG ID 12 PG GUID 35(EXPERT_MODEL_PARALLEL_GROUP) Rank 2] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xeba71106c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x224 (0xeba711123574 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x4c (0xeba711cbe0fc in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x74 (0xeba711cdd404 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop() + 0x770 (0xeba711ce4460 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #5: c10d::ProcessGroupNCCL::Watchdog::run() + 0xc8 (0xeba711ce5e18 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xe1ae0 (0xeba743be1ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #7: <unknown function> + 0x8595c (0xeba744e2595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #8: <unknown function> + 0xebb0c (0xeba744e8bb0c in /lib/aarch64-linux-gnu/libc.so.6)

Exception raised from run at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2063 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xeba71106c700 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xebf100 (0xeba711c9f100 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::Watchdog::run() + 0x474 (0xeba711ce61c4 in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xe1ae0 (0xeba743be1ae0 in /lib/aarch64-linux-gnu/libstdc++.so.6)
frame #4: <unknown function> + 0x8595c (0xeba744e2595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #5: <unknown function> + 0xebb0c (0xeba744e8bb0c in /lib/aarch64-linux-gnu/libc.so.6)
 (most recent call first):
  <no Python frame>
Fatal Python error: 
  <no Python frame>

Aborted
Thread 0xFatal Python error: 

Thread 0x0000e511de74f180AbortedThread 0x0000eddb237ef180 (most recent call first):
  <no Python frame>


0000ec01737ef180 (most recent call first):

Thread 0xThread 0x (most recent call first):
  File 0000e511def5f180  <no Python frame>
" (most recent call first):

/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/tensor_parallel/mappings.py0000eba19d72f180  <no Python frame>
Thread 0x" (most recent call first):

0000ec0172fdf180, line   <no Python frame>
Thread 0x (most recent call first):
439
0000e5122c99f180  <no Python frame>
 in Thread 0x (most recent call first):

forward0000eba19df3f180  File Thread 0x
 (most recent call first):
"  File   File /usr/lib/python3.12/threading.py"0000ec0173fff180""/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/autograd/function.py (most recent call first):
/wireless/minyusong/TE2/TransformerEngine/transformer_engine/pytorch/router.py, line "  <no Python frame>
"359, line 
, line  in 581Thread 0x260wait in 0000ec018082f180 in 
apply (most recent call first):
backward  File 
  <no Python frame>

"  File 
  File /usr/lib/python3.12/threading.py"Thread 0x""/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/tensor_parallel/mappings.py0000ec018103f180/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/autograd/function.py, line " (most recent call first):
"655, line   File , line  in 458"315wait in /usr/lib/python3.12/threading.py in 
backward"apply  File 
, line 
  File "359
"/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tqdm/_monitor.py in Thread 0x/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/autograd/function.py"wait", line 
0000eba19e74f180, line 60  File  (most recent call first):
315 in "  <no Python frame>
 in run/usr/lib/python3.12/threading.py
apply
"Thread 0x
  File , line 
"6550000eba19ef5f180Thread 0x/usr/lib/python3.12/threading.py in  (most recent call first):
0000eddb23fff180" (most recent call first):
wait  <no Python frame>
, line   File 

1073"  File Thread 0x in /usr/lib/python3.12/threading.py"0000eba1ec99f180_bootstrap_inner"/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tqdm/_monitor.py (most recent call first):

, line "  File   File 359, line "" in 60/usr/lib/python3.12/threading.py/usr/lib/python3.12/threading.pywait in ""
run, line , line   File 
3591030"  File  in  in /usr/lib/python3.12/threading.py"wait_bootstrap"/usr/lib/python3.12/threading.py

, line "  File 
655, line "Thread 0x in 1073/usr/lib/python3.12/threading.py0000e512b2fdf180wait in " (most recent call first):

_bootstrap_inner, line   File   File 
655""  File  in /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tqdm/_monitor.py/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py"wait""/usr/lib/python3.12/threading.py
, line , line "  File 6073, line " in  in 1030/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tqdm/_monitor.pyrun_recv_msg in "

_bootstrap, line   File 
60  File "
 in "/usr/lib/python3.12/threading.pyThread 0xrun/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py"0000ec020cf9f180
", line  (most recent call first):
  File , line 1073"228 in /usr/lib/python3.12/threading.py in _bootstrap_inner  File "_read_thread
", line 
  File /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py1073  File "" in "/usr/lib/python3.12/threading.py, line _bootstrap_inner/usr/lib/python3.12/threading.py"73
", line  in   File , line 1030_recv_msg"1010 in 
/usr/lib/python3.12/threading.py in _bootstrap  File "run
", line 

  File /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py1030Thread 0x"0000eddbaefdf180" in /usr/lib/python3.12/threading.py (most recent call first):
, line _bootstrap"  File 228
, line " in 
1073/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py_read_threadThread 0x in "
0000eba272fdf180_bootstrap_inner, line   File  (most recent call first):

73"  File  in /usr/lib/python3.12/threading.py"_recv_msg  File "/usr/lib/python3.12/threading.py
, line ""  File 1010/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py, line " in "1030/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.pyrun, line  in "
73_bootstrap, line   File  in 
228"_recv_msg
 in Thread 0x/usr/lib/python3.12/threading.py
_read_thread0000e51783cc44e0"  File 
 (most recent call first):
, line "  File   File "1073/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/_inductor/compile_worker/subproc_pool.py"/usr/lib/python3.12/threading.py in "/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py"_bootstrap_inner, line ", line 
228, line 1010  File  in 1797 in "_read_thread in run/usr/lib/python3.12/threading.py
_distributed_excepthook
"  File 
  File , line ""1030/usr/lib/python3.12/threading.py/usr/lib/python3.12/threading.py in ""_bootstrap, line , line 
10101073
 in  in Thread 0xrun
Extension modules: _bootstrap_inner0000ec02fa46f180
numpy._core._multiarray_umath
 (most recent call first):
  File   File   File ", ""/usr/lib/python3.12/threading.pynumpy.linalg._umath_linalg/usr/lib/python3.12/threading.py/usr/lib/python3.12/threading.py"", ", line , line torch._C, line 10731030,  in 359 in torch._C._dynamo.autograd_compiler_bootstrap in _bootstrap_inner, 
wait
torch._C._dynamo.eval_frame

  File , Thread 0x  File "torch._C._dynamo.guards0000ede0897d44e0"/usr/lib/python3.12/threading.py,  (most recent call first):
/usr/lib/python3.12/queue.py"torch._C._dynamo.utils  File ", line , ", line 1030torch._C._fft/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.py180 in , " in _bootstraptorch._C._linalg, line get
, 189

torch._C._nested in   File Thread 0x, custom_backward"0000eba7451244e0torch._C._nn
/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py (most recent call first):
,   File "  File torch._C._sparse", line ", /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.py269/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.pytorch._C._special" in ", line _run, line 487
189 in   File  in backward_step"custom_backward
/cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py
  File ""  File , /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.py, line "google._upb._message"244/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.py, line  in ", 651run, line numpy.random._common in 
487, forward_backward_no_pipelining  File  in numpy.random.bit_generator
"backward_step,   File /usr/lib/python3.12/threading.py
numpy.random._bounded_integers""  File , /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.py, line "numpy.random._pcg64"1073/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/core/pipeline_parallel/schedules.py, , line  in "numpy.random._generator1652_bootstrap_inner, line ,  in 
651numpy.random._mt19937train_step  File  in 
, "forward_backward_no_pipelining  File numpy.random._philox/usr/lib/python3.12/threading.py
", "  File /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.pynumpy.random._sfc64, line "", 1030/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.pynumpy.random.mtrand, line  in "2795_bootstrap, line  in 
1652train
 in 
, Thread 0xtrain_step  File yaml._yaml0000ec06e1c544e0
" (most recent call first):
  File /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.py  File "", "/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.pymarkupsafe._speedups, line /cpfs01/laiqingsi/project/moe/venv_megatron/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py"1031, line ",  in 2795, line regex._regexpretrain in 1797
, train in   File psutil._psutil_linux
_distributed_excepthook"  File 
, /cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/pretrain_gpt.py"charset_normalizer.md"/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/megatron/training/training.py, line , "342requests.packages.charset_normalizer.md in <module>, , line 
requests.packages.chardet.md1031 in pretrain
,   File PIL._imaging"/cpfs01/laiqingsi/Megatron-LM-core_v0.16.1/pretrain_gpt.py
Extension modules: ", numpy._core._multiarray_umath, line msgpack._cmsgpack342,  in , numpy.linalg._umath_linalg
Extension modules: <module>sentencepiece._sentencepiecenumpy._core._multiarray_umath, 
, torch._C, , cuda_utilsnumpy.linalg._umath_linalgtorch._C._dynamo.autograd_compiler, , , __triton_launchertorch._Ctorch._C._dynamo.eval_frame (total: , , 35torch._C._dynamo.autograd_compilertorch._C._dynamo.guards), , 
torch._C._dynamo.eval_frametorch._C._dynamo.utils
Extension modules: , , numpy._core._multiarray_umathtorch._C._dynamo.guardstorch._C._fft, , , torch._C._dynamo.utilsnumpy.linalg._umath_linalgtorch._C._linalg, , , torch._C._ffttorch._C._nestedtorch._C, , , torch._C._linalgtorch._C._nntorch._C._dynamo.autograd_compiler, , , torch._C._nestedtorch._C._sparsetorch._C._dynamo.eval_frame, , , torch._C._nntorch._C._specialtorch._C._dynamo.guards, , torch._C._sparsetorch._C._dynamo.utils, , torch._C._specialtorch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, google._upb._message, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._generator, numpy.random._mt19937, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand, yaml._yaml, markupsafe._speedups, regex._regex, , google._upb._message, psutil._psutil_linuxgoogle._upb._message, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._generator, , , charset_normalizer.mdnumpy.random._commonnumpy.random._mt19937, , , numpy.random.bit_generatorrequests.packages.charset_normalizer.mdnumpy.random._philox, , , numpy.random._bounded_integersrequests.packages.chardet.mdnumpy.random._sfc64, , numpy.random._pcg64numpy.random.mtrand, , numpy.random._generatorPIL._imaging, numpy.random._mt19937, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand, yaml._yaml, msgpack._cmsgpack, markupsafe._speedups, sentencepiece._sentencepiece, regex._regex, psutil._psutil_linux, , charset_normalizer.mdcuda_utils, requests.packages.charset_normalizer.md, requests.packages.chardet.md, __triton_launcher (total: 35, )PIL._imaging
, msgpack._cmsgpack, sentencepiece._sentencepiece, cuda_utils, yaml._yaml, __triton_launcher (total: 35)
, markupsafe._speedups, regex._regex, psutil._psutil_linux, charset_normalizer.md, requests.packages.charset_normalizer.md, requests.packages.chardet.md, PIL._imaging, msgpack._cmsgpack, sentencepiece._sentencepiece, cuda_utils, __triton_launcher (total: 35)
[2026-05-12 10:21:40,129] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 1090217
[2026-05-12 10:21:40,129] [INFO] [launch.py:335:sigkill_handler] Killing subprocess 1090218
