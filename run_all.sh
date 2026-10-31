#!/usr/bin/env bash
# Each model starts/stops sequentially on a single L4. Existing GGUF downloads are required,
# or explicitly pass --download when invoking this script.
set -euo pipefail
cd "$(dirname "$0")"
for model in qwen35_fp8_vllm gemma4_12b_qat_gguf hyperclovax_14b_q4_gguf qwen38_27b_q4_gguf; do
    echo "===== $model ====="
    python run_model.py --profile "$model" "$@"
done
python compare_results.py results/qwen35_fp8_vllm results/gemma4_12b_qat_gguf results/hyperclovax_14b_q4_gguf results/qwen38_27b_q4_gguf --out comparison.md
echo "Complete: comparison.md"
