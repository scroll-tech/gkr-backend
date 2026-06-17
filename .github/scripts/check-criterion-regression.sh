#!/usr/bin/env bash
set -euo pipefail

base_sha="${1:?base sha is required}"
head_sha="${2:?head sha is required}"

bench="${GKR_PERF_BENCH:-ceno_batched_main}"
baseline="${GKR_PERF_BASELINE:-ci-main}"
noise_threshold="${GKR_PERF_NOISE_THRESHOLD:-0.15}"
output_file="${RUNNER_TEMP:-/tmp}/criterion-${bench}-comparison.log"

export RUST_MIN_STACK="${RUST_MIN_STACK:-33554432}"
export GKR_CENO_BENCH_SCALE="${GKR_CENO_BENCH_SCALE:-tiny}"
export GKR_CENO_BENCH_SAMPLES="${GKR_CENO_BENCH_SAMPLES:-10}"

fetch_commit() {
  local remote_url="$1"
  local sha="$2"

  if git rev-parse --verify --quiet "${sha}^{commit}" >/dev/null; then
    return
  fi

  git fetch --no-tags --depth=1 "$remote_url" "$sha"
}

if [[ -n "${BASE_REPO_URL:-}" ]]; then
  fetch_commit "$BASE_REPO_URL" "$base_sha"
fi

if [[ -n "${HEAD_REPO_URL:-}" ]]; then
  fetch_commit "$HEAD_REPO_URL" "$head_sha"
fi

git checkout --detach "$base_sha"
cargo bench -p sumcheck --bench "$bench" -- --save-baseline "$baseline"

git checkout --detach "$head_sha"
cargo bench -p sumcheck --bench "$bench" -- \
  --baseline "$baseline" \
  --noise-threshold "$noise_threshold" \
  | tee "$output_file"

if grep -q "Performance has regressed" "$output_file"; then
  echo "::error::Criterion reported a sumcheck performance regression."
  awk '
    /^Benchmarking / { benchmark = $0 }
    /^[[:space:]]*change:/ { change = $0 }
    /Performance has regressed/ {
      print benchmark
      print change
      print $0
      print ""
    }
  ' "$output_file"
  exit 1
fi

echo "No Criterion performance regression detected for ${bench}."
