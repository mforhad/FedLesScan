#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=300
clients_per_round=175
allowed_stragglers=175
accuracy_threshold=0.99
rounds=40
dataset_name="femnist"
client_timeout=130

base_out_dir="$root_directory/out/real_world/expo"
config_dir="$script_dir/$dataset_name-$n_clients-$clients_per_round.yaml"
echo $base_out_dir
# shellcheck disable=SC2034
for straggler_percent in  0.7 ; do
  
  
  python -m fedless.controller.scripts \
    -d "$dataset_name" \
    -s "fedlesscan" \
    -c "$config_dir" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/$dataset_name-enhanced-$straggler_percent" \
    --aggregate-online \
    --rounds "$rounds" \
    --timeout "$client_timeout" \
    --simulate-stragglers "$straggler_percent"
  
  sleep 1

  python -m fedless.controller.scripts \
    -d "$dataset_name" \
    -s "fedavg" \
    -c "$config_dir" \
    --clients "$n_clients" \
    --clients-in-round "$clients_per_round" \
    --stragglers "$allowed_stragglers" \
    --max-accuracy "$accuracy_threshold" \
    --out "$base_out_dir/$dataset_name-$straggler_percent" \
    --aggregate-online \
    --rounds "$rounds" \
    --timeout "$client_timeout" \
    --simulate-stragglers "$straggler_percent" 


  
  # sleep 1

  # python -m fedless.controller.scripts \
  #   -d "$dataset_name" \
  #   -s "fedprox" \
  #   -c "$config_dir" \
  #   --clients "$n_clients" \
  #   --clients-in-round "$clients_per_round" \
  #   --stragglers "$allowed_stragglers" \
  #   --max-accuracy "$accuracy_threshold" \
  #   --out "$base_out_dir/$dataset_name-prox-$straggler_percent" \
  #   --rounds "$rounds" \
  #   --aggregate-online \
  #   --timeout "$client_timeout" \
  #   --simulate-stragglers "$straggler_percent" \
  #   --mu 0.001

done

exit 0

# done
