#!/bin/bash
set -euo pipefail

# ============================================================
# Launch N IC replica nodes, each on unique loopback IP
# ============================================================

TESTNET_DIR="/workspace/ic-testnet"
IC_DIR="/workspace/ic"
REPLICA="$IC_DIR/bazel-bin/rs/replica/replica"
REPLICA_DIR=$(dirname $(readlink -f "$REPLICA"))
export LD_LIBRARY_PATH="$REPLICA_DIR:${LD_LIBRARY_PATH:-}"
NUM_NODES=${1:-28}
REPLICA_VERSION="0.0.1"
REGISTRY_STORE="$TESTNET_DIR/state/ic_registry_local_store"

# Same ports for all — unique IPs differentiate
P2P_PORT=4100
XNET_PORT=4200
HTTP_PORT=18080
METRICS_PORT=9090

# Kill any existing replicas
echo "=== Stopping existing replicas ==="
pkill -f "replica.*replica-version" 2>/dev/null || true
sleep 1

# Create log directory
mkdir -p "$TESTNET_DIR/logs"

echo "=== Launching $NUM_NODES replica nodes ==="

for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_IP="127.0.0.$((i + 1))"
    NODE_DIR="$TESTNET_DIR/state/node-$i"
    DATA_DIR="$TESTNET_DIR/data/node-$i"

    # Create per-node data directories
    mkdir -p "$DATA_DIR/state" "$DATA_DIR/consensus_pool" "$DATA_DIR/backup"

    NODE_ID=$(cat "$NODE_DIR/derived_node_id")

    CONFIG=$(cat <<EOCFG
{
    subnet_id: 0,
    transport: {
        node_ip: "${NODE_IP}",
        listening_port: ${P2P_PORT},
    },
    initial_ipv4_config: { public_address: "", public_gateway: "" },
    domain: "",
    registry_client: { local_store: "${REGISTRY_STORE}" },
    state_manager: { state_root: "${DATA_DIR}/state" },
    artifact_pool: {
        consensus_pool_path: "${DATA_DIR}/consensus_pool",
        ingress_pool_max_count: 9223372036854775807,
        ingress_pool_max_bytes: 9223372036854775807,
        backup: { spool_path: "${DATA_DIR}/backup/", retention_time_secs: 3600, purging_interval_secs: 3600 }
    },
    crypto: { crypto_root: "${NODE_DIR}/crypto", csp_vault_type: "in_replica" },
    scheduler: { scheduler_cores: ${SCHEDULER_CORES:-4}, max_instructions_per_round: 26843545600, max_instructions_per_message: 5368709120 },
    hypervisor: {},
    tracing: {},
    http_handler: { listen_addr: "${NODE_IP}:${HTTP_PORT}" },
    metrics: {
        exporter: { http: "${NODE_IP}:${METRICS_PORT}" },
        connection_read_timeout_seconds: 300, max_concurrent_requests: 50, request_timeout_seconds: 30
    },
    logger: { level: "info", format: "text_full", block_on_overflow: false },
    orchestrator_logger: { level: "info", format: "text_full", block_on_overflow: false },
    csp_vault_logger: { level: "info", format: "text_full", block_on_overflow: false },
    message_routing: {},
    malicious_behavior: {
       allow_malicious_behavior: false, maliciously_seg_fault: false,
       malicious_flags: {
         maliciously_propose_equivocating_blocks: false, maliciously_propose_empty_blocks: false,
         maliciously_finalize_all: false, maliciously_notarize_all: false,
         maliciously_tweak_dkg: false, maliciously_certify_invalid_hash: false,
         maliciously_malfunctioning_xnet_endpoint: false, maliciously_disable_execution: false,
         maliciously_corrupt_own_state_at_heights: [], maliciously_disable_ingress_validation: false,
         maliciously_corrupt_idkg_dealings: false, maliciously_alter_certified_hash: false,
         maliciously_alter_state_sync_chunk_sending_side: false,
       },
    },
    firewall: {
        config_file: "/dev/null", file_template: "",
        ipv4_tcp_rule_template: "", ipv4_udp_rule_template: "",
        ipv6_tcp_rule_template: "", ipv6_udp_rule_template: "",
        ipv4_user_output_rule_template: "", ipv6_user_output_rule_template: "",
        default_rules: [], tcp_ports_for_node_whitelist: [], udp_ports_for_node_whitelist: [],
        ports_for_http_adapter_blacklist: [], max_simultaneous_connections_per_ip_address: 0,
    },
    boundary_node_firewall: {
        config_file: "/dev/null", file_template: "",
        ipv4_tcp_rule_template: "", ipv4_udp_rule_template: "",
        ipv6_tcp_rule_template: "", ipv6_udp_rule_template: "",
        default_rules: [], max_simultaneous_connections_per_ip_address: 0,
    },
    registration: { pkcs11_keycard_transport_pin: "358138" },
    nns_registry_replicator: { poll_delay_duration_ms: 5000 },
    adapters_config: {
        bitcoin_testnet_uds_path: "/tmp/bitcoin_uds_${i}",
        https_outcalls_uds_path: "/tmp/https_outcalls_${i}",
    },
    bitcoin_payload_builder_config: {},
}
EOCFG
)

    echo "Starting node $i @ ${NODE_IP} (ID: ${NODE_ID:0:10}...)"

    $REPLICA \
        --replica-version "$REPLICA_VERSION" \
        --config-literal "$CONFIG" \
        > "$TESTNET_DIR/logs/node-${i}.log" 2>&1 &

done

echo ""
echo "=== All $NUM_NODES nodes launched ==="
echo "Logs: $TESTNET_DIR/logs/"
echo ""
echo "Collect metrics: bash $TESTNET_DIR/collect_metrics.sh"
echo "Stop: pkill -f 'replica.*replica-version'"
