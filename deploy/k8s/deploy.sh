#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 CONTEXT IMAGE CONFIG_ENV SECRET_ENV [NAMESPACE]" >&2
  exit 2
fi
context=$1
image=$2
config_file=$3
secret_file=$4
namespace=${5:-hdfc-support}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v kubectl >/dev/null
[[ -f "$config_file" && -f "$secret_file" ]] || { echo "Missing environment file." >&2; exit 2; }
kube=(kubectl --context "$context" --namespace "$namespace")
"${kube[@]}" create namespace "$namespace" --dry-run=client -o yaml | "${kube[@]}" apply -f -
"${kube[@]}" create configmap hdfc-support-config --from-env-file="$config_file" --dry-run=client -o yaml |
  "${kube[@]}" apply --server-side --field-manager=hdfc-deploy -f -
# Server-side apply avoids copying secret contents into last-applied annotations.
"${kube[@]}" create secret generic hdfc-support-secrets --from-env-file="$secret_file" --dry-run=client -o yaml |
  "${kube[@]}" apply --server-side --field-manager=hdfc-deploy -f -
kubectl set image --local -f "$script_dir/deployment.yaml" "api=$image" "ledger-init=$image" -o yaml |
  "${kube[@]}" apply -f -
"${kube[@]}" apply -f "$script_dir/service.yaml" -f "$script_dir/scaling.yaml"
# Environment changes must restart pods even if the image did not change.
"${kube[@]}" rollout restart deployment/hdfc-support
"${kube[@]}" rollout status deployment/hdfc-support --timeout=300s
