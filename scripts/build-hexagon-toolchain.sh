#!/usr/bin/env bash
# Builds the Hexagon toolchain Docker image used for compile verification
# (see docker/README.md). Heavy build — full LLVM/clang from source, expect
# 30-90+ minutes and several GB of disk on first run. Run once; cached after.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build \
  --target toolchain \
  -t cuda2ripple-hexagon-toolchain:latest \
  -f docker/hexagon-toolchain.Dockerfile \
  .

echo ""
echo "Built cuda2ripple-hexagon-toolchain:latest"
echo "Verify: docker run --rm cuda2ripple-hexagon-toolchain:latest /opt/hexagon/bin/clang --version"
