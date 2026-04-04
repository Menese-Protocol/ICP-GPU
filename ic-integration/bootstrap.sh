#!/bin/bash
set -euo pipefail

# ============================================================
# Bootstrap an N-node IC test subnet on localhost
# Each node gets unique IP 127.0.0.(N+1)
# No NNS required — just real consensus with real BLS crypto
# ============================================================

TESTNET_DIR="/workspace/ic-testnet"
IC_DIR="/workspace/ic"
IC_PREP="$IC_DIR/bazel-bin/rs/prep/ic-prep"
REPLICA="$IC_DIR/bazel-bin/rs/replica/replica"
NUM_NODES=${1:-28}
REPLICA_VERSION="0.0.1"

# Each node gets unique IP, same ports
P2P_PORT=4100
XNET_PORT=4200
HTTP_PORT=18080
METRICS_PORT=9090

echo "=== IC Testnet Bootstrap: ${NUM_NODES}-node subnet ==="
echo "Replica: $REPLICA"
echo "ic-prep: $IC_PREP"

# Clean previous state
rm -rf "$TESTNET_DIR/state" "$TESTNET_DIR/data" "$TESTNET_DIR/logs"
mkdir -p "$TESTNET_DIR/state"

# Build --node arguments for ic-prep
# Each node: unique IP 127.0.0.(i+1), same ports
NODE_ARGS=""
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_IP="127.0.0.$((i + 1))"
    NODE_ARGS="$NODE_ARGS --node 'idx:${i},subnet_idx:0,xnet_api:\"${NODE_IP}:${XNET_PORT}\",public_api:\"${NODE_IP}:${HTTP_PORT}\"'"
done

echo ""
echo "=== Running ic-prep ==="
echo "Nodes: $NUM_NODES (IPs: 127.0.0.1-127.0.0.${NUM_NODES})"

# Run ic-prep to generate DKG keys and registry
eval "$IC_PREP \
    --working-dir '$TESTNET_DIR/state' \
    --replica-version '$REPLICA_VERSION' \
    --allow-empty-update-image \
    --dkg-interval-length 10 \
    --nns-subnet-index 0 \
    --use-specified-ids-allocation-range \
    --provisional-whitelist '$TESTNET_DIR/provisional_whitelist.json' \
    $NODE_ARGS"

echo ""
echo "=== ic-prep completed ==="
echo "Generated $(ls -d $TESTNET_DIR/state/node-* | wc -l) node directories"

REGISTRY_STORE=$(find "$TESTNET_DIR/state" -name "ic_registry_local_store" -type d | head -1)
echo "Registry store: $REGISTRY_STORE"

echo ""
echo "=== Bootstrap complete ==="
echo "Next: run launch.sh $NUM_NODES to start all replicas"
