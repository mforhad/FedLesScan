#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"

n_clients=100
clients_per_round=10
allowed_stragglers=5
accuracy_threshold=0.99

# shellcheck disable=SC2034
for curr_repeat in {1..3}; do
  python3 -m fedless.benchmark.scripts \
    -d "mnist" \
    -s "fedless" \
    -c fedless-mnist-flower-comp.yaml \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$root_directory/out/fedless-mnist-flower" \
    --tum-proxy
  sleep 600
done
