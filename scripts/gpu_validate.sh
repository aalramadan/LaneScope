#!/usr/bin/env bash
set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/venv.sh"
out="${1:-check-$(date +%Y-%m-%d)}"

if ! command -v g++ >/dev/null; then
  echo "g++ is missing. Install it with: sudo apt install g++" >&2
  exit 1
fi
lanescope_install "[gpu]"
"$venv/bin/lanescope" gpu-info

# The CUDA wheels must not be newer than the CUDA version of the driver.
cuda="$("$venv/bin/lanescope" gpu-info --cuda-version)"
major="${cuda%%.*}"
minor="${cuda#*.}"
if [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt 4 ]; }; then
  echo "The driver supports CUDA $cuda. LaneScope needs CUDA 12.4 or later, so update the NVIDIA driver." >&2
  exit 1
fi
if [ "$major" -eq 12 ] && [ "$minor" -lt 9 ]; then
  # Some minor versions, such as 12.7, have no wheels, so the limit is an upper bound.
  limit=">=12.4,<12.$((minor + 1))"
  "$venv/bin/pip" install --quiet \
    "nvidia-cuda-nvrtc-cu12$limit" "nvidia-cuda-nvcc-cu12$limit" \
    "nvidia-cuda-sanitizer-api-cu12$limit" "nvidia-cuda-runtime-cu12$limit"
  # A CUDA toolkit on this machine would take precedence and may be newer than the driver.
  export LANESCOPE_USE_WHEELS=1
fi

# Nsight Compute ships apart from the CUDA wheels. Without one on this machine, the
# command fetches the version that suits the driver, a download of 300 to 600 MB.
# The check still records the kernels when the download fails.
if ! "$venv/bin/lanescope" get-ncu; then
  echo "The Nsight Compute download failed, so the check runs without it." \
    "Run $venv/bin/lanescope get-ncu or install Nsight Compute, then run this script again." >&2
fi

"$venv/bin/lanescope" validate --out "$out"
echo "Summary: $out/validation.md"
