#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=270
clients_per_round=100
allowed_stragglers=85
accuracy_threshold=0.9
rounds=30
# straggler_percent=0.2

base_out_dir="$root_directory/out/controlled-expo"

# shellcheck disable=SC2034
for straggler_percent in 0.1 0.2 0.3; do

  python -m fedless.controller.scripts \
    -d "speech" \
    -s "fedlesscan" \
    -c "$script_dir/speech-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/speech-enhanced-"$straggler_percent"" \
    --rounds "$rounds" \
    --timeout 90000 \
    --mock \
    --simulate-stragglers "$straggler_percent"
  sleep 2
  
  python -m fedless.controller.scripts \
    -d "speech" \
    -s "fedavg" \
    -c "$script_dir/speech-demo.yaml" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/speech-"$straggler_percent"" \
    --rounds "$rounds" \
    --timeout 90000 \
    --mock \
    --simulate-stragglers "$straggler_percent"
  
  sleep 2

done
# python -m fedless.controller.scripts \
#   -d "speech" \
#   -s "fedprox" \
#   -c "$script_dir/speech-demo.yaml" \
#   --clients "$n_clients" \
#   --clients-in-round "$clients_per_round" \
#   --stragglers "$allowed_stragglers" \
#   --max-accuracy "$accuracy_threshold" \
#   --out "$root_directory/out/controlled/speech-d-prx-0.1-"$straggler_percent"" \
#   --rounds "$rounds" \
#   --timeout 90000 \
#   --mock \
#   --simulate-stragglers "$straggler_percent" \
#   --mu 0.1 &

# done
