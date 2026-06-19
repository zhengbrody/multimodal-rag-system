#!/bin/bash
# Run JMeter load test
# Usage: ./run_load_test.sh [target_host] [target_port]

set -euo pipefail

TARGET_HOST="${1:-localhost}"
TARGET_PORT="${2:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PLAN="${SCRIPT_DIR}/test_plan.jmx"
RESULTS_DIR="${SCRIPT_DIR}/results"
RESULTS_FILE="${RESULTS_DIR}/results.jtl"
REPORT_DIR="${RESULTS_DIR}/report"

# Verify JMeter is installed
if ! command -v jmeter &> /dev/null; then
    echo "ERROR: JMeter is not installed or not in PATH."
    echo "Install it via: brew install jmeter (macOS) or download from https://jmeter.apache.org/"
    exit 1
fi

# Verify test plan exists
if [ ! -f "${TEST_PLAN}" ]; then
    echo "ERROR: Test plan not found at ${TEST_PLAN}"
    exit 1
fi

# Clean previous results
if [ -d "${RESULTS_DIR}" ]; then
    echo "Cleaning previous results..."
    rm -rf "${RESULTS_DIR}"
fi
mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "  JMeter Load Test - RAG System"
echo "============================================"
echo "Target:     ${TARGET_HOST}:${TARGET_PORT}"
echo "Test Plan:  ${TEST_PLAN}"
echo "Results:    ${RESULTS_DIR}"
echo "============================================"
echo ""

# Run JMeter in non-GUI mode
jmeter -n \
    -t "${TEST_PLAN}" \
    -l "${RESULTS_FILE}" \
    -e -o "${REPORT_DIR}" \
    -Jtarget.host="${TARGET_HOST}" \
    -Jtarget.port="${TARGET_PORT}" \
    -Jquestions.csv="${SCRIPT_DIR}/questions.csv"

echo ""
echo "============================================"
echo "  Test Complete"
echo "============================================"
echo "Raw results:   ${RESULTS_FILE}"
echo "HTML report:   ${REPORT_DIR}/index.html"
echo ""

# Print summary if results file exists
if [ -f "${RESULTS_FILE}" ]; then
    TOTAL=$(wc -l < "${RESULTS_FILE}" | tr -d ' ')
    TOTAL=$((TOTAL - 1))  # subtract header line
    ERRORS=$(awk -F',' 'NR>1 && $8=="false" {count++} END {print count+0}' "${RESULTS_FILE}")
    echo "Total requests: ${TOTAL}"
    echo "Failed requests: ${ERRORS}"
    if [ "${TOTAL}" -gt 0 ]; then
        ERROR_RATE=$(awk "BEGIN {printf \"%.2f\", (${ERRORS}/${TOTAL})*100}")
        echo "Error rate:     ${ERROR_RATE}%"
    fi
fi

echo ""
echo "Open the HTML report: open ${REPORT_DIR}/index.html"
