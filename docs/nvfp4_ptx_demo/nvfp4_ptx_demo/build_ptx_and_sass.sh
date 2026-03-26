#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/generated"
mkdir -p "${OUTPUT_DIR}"

if ! command -v nvcc >/dev/null 2>&1; then
	echo "nvcc not found. Install a recent CUDA Toolkit first." >&2
	exit 1
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
	echo "cuobjdump not found. Install CUDA Binary Utilities." >&2
	exit 1
fi

if ! command -v nvdisasm >/dev/null 2>&1; then
	echo "nvdisasm not found. Install CUDA Binary Utilities." >&2
	exit 1
fi

NVCC_COMMON_FLAGS=(
	-std=c++17
	-lineinfo
	-Xptxas=-v
	--keep
)

build_one() {
	local source_file="$1"
	local architecture_name="$2"
	local output_stem="$3"

	nvcc "${NVCC_COMMON_FLAGS[@]}" \
		--generate-code "arch=compute_${architecture_name},code=compute_${architecture_name}" \
		-ptx "${SCRIPT_DIR}/${source_file}" \
		-o "${OUTPUT_DIR}/${output_stem}.ptx"

	nvcc "${NVCC_COMMON_FLAGS[@]}" \
		--generate-code "arch=compute_${architecture_name},code=sm_${architecture_name}" \
		-cubin "${SCRIPT_DIR}/${source_file}" \
		-o "${OUTPUT_DIR}/${output_stem}.cubin"

	nvdisasm -g "${OUTPUT_DIR}/${output_stem}.cubin" > "${OUTPUT_DIR}/${output_stem}.nvdisasm.sass"
	cuobjdump --dump-sass "${OUTPUT_DIR}/${output_stem}.cubin" > "${OUTPUT_DIR}/${output_stem}.cuobjdump.sass"
}

build_one "nvfp4_mma_sync_minimal.cu" "120a" "nvfp4_mma_sync_minimal"
build_one "tcgen05_mma_nvf4_minimal.cu" "100a" "tcgen05_mma_nvf4_minimal"

echo "Artifacts written to ${OUTPUT_DIR}"
