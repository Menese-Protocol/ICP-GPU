#!/bin/bash
# Collect crypto timing metrics from all 7 IC replica nodes
# Uses built-in Prometheus metrics at /metrics endpoint

echo "=== IC Testnet Crypto Metrics Report ==="
echo "Time: $(date)"
echo ""

# Collect from node 0 (representative — all nodes do similar work)
METRICS=$(curl -s http://127.0.0.1:9090/metrics 2>/dev/null)

if [ -z "$METRICS" ]; then
    echo "ERROR: Cannot reach node 0 metrics at :9100"
    exit 1
fi

# Extract key crypto metrics
echo "=== Threshold Signature Verification (BLS - GPU target) ==="
echo "$METRICS" | grep "crypto_duration_seconds" | grep "threshold_signature.*verify_threshold_sig_share" | grep "_sum\|_count"

echo ""
echo "=== Multi-Signature Verification ==="
echo "$METRICS" | grep "crypto_duration_seconds" | grep "multi_signature.*verify_multi_sig_individual" | grep "_sum\|_count"

echo ""
echo "=== Threshold Signature Signing ==="
echo "$METRICS" | grep "crypto_duration_seconds" | grep "threshold_signature.*sign_threshold" | grep "_sum\|_count"

echo ""
echo "=== Threshold Signature Combine ==="
echo "$METRICS" | grep "crypto_duration_seconds" | grep "threshold_signature.*combine" | grep "_sum\|_count"

echo ""
echo "=== DKG Verification ==="
echo "$METRICS" | grep "crypto_duration_seconds" | grep "ni_dkg.*verify_dealing" | grep "_sum\|_count"

echo ""
echo "=== Certification Processing ==="
echo "$METRICS" | grep "artifact_manager_client_processing" | grep "certification" | grep "_sum\|_count"

echo ""
echo "=== Consensus Heights ==="
echo "$METRICS" | grep -i "consensus.*height\|finalized_height\|certified_height\|notarized_height" | grep -v "bucket" | head -10

echo ""
echo "=== Per-call Averages ==="
# Calculate averages
VERIFY_SUM=$(echo "$METRICS" | grep 'crypto_duration_seconds_sum{domain="threshold_signature",method_name="verify_threshold_sig_share"' | awk '{print $2}')
VERIFY_COUNT=$(echo "$METRICS" | grep 'crypto_duration_seconds_count{domain="threshold_signature",method_name="verify_threshold_sig_share"' | awk '{print $2}')
MULTI_SUM=$(echo "$METRICS" | grep 'crypto_duration_seconds_sum{domain="multi_signature",method_name="verify_multi_sig_individual"' | awk '{print $2}')
MULTI_COUNT=$(echo "$METRICS" | grep 'crypto_duration_seconds_count{domain="multi_signature",method_name="verify_multi_sig_individual"' | awk '{print $2}')
COMBINE_SUM=$(echo "$METRICS" | grep 'crypto_duration_seconds_sum{domain="threshold_signature",method_name="combine_threshold_sig_shares"' | awk '{print $2}')
COMBINE_COUNT=$(echo "$METRICS" | grep 'crypto_duration_seconds_count{domain="threshold_signature",method_name="combine_threshold_sig_shares"' | awk '{print $2}')

if [ -n "$VERIFY_COUNT" ] && [ "$VERIFY_COUNT" != "0" ]; then
    AVG=$(echo "scale=6; $VERIFY_SUM / $VERIFY_COUNT * 1000" | bc)
    echo "verify_threshold_sig_share: ${AVG}ms avg (${VERIFY_COUNT} calls, ${VERIFY_SUM}s total)"
fi
if [ -n "$MULTI_COUNT" ] && [ "$MULTI_COUNT" != "0" ]; then
    AVG=$(echo "scale=6; $MULTI_SUM / $MULTI_COUNT * 1000" | bc)
    echo "verify_multi_sig_individual: ${AVG}ms avg (${MULTI_COUNT} calls, ${MULTI_SUM}s total)"
fi
if [ -n "$COMBINE_COUNT" ] && [ "$COMBINE_COUNT" != "0" ]; then
    AVG=$(echo "scale=6; $COMBINE_SUM / $COMBINE_COUNT * 1000" | bc)
    echo "combine_threshold_sig_shares: ${AVG}ms avg (${COMBINE_COUNT} calls, ${COMBINE_SUM}s total)"
fi

echo ""
echo "=== All Nodes Summary ==="
for i in $(seq 0 6); do
    NODE_IP="127.0.0.$((i + 1))"
    N_METRICS=$(curl -s http://$NODE_IP:9090/metrics 2>/dev/null)
    V_COUNT=$(echo "$N_METRICS" | grep 'crypto_duration_seconds_count{domain="threshold_signature",method_name="verify_threshold_sig_share"' | awk '{print $2}')
    V_SUM=$(echo "$N_METRICS" | grep 'crypto_duration_seconds_sum{domain="threshold_signature",method_name="verify_threshold_sig_share"' | awk '{print $2}')
    if [ -n "$V_COUNT" ] && [ "$V_COUNT" != "0" ]; then
        AVG=$(echo "scale=3; $V_SUM / $V_COUNT * 1000" | bc)
        echo "  Node $i: ${V_COUNT} verify calls, ${AVG}ms avg"
    else
        echo "  Node $i: no data"
    fi
done
