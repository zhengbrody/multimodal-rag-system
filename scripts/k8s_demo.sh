#!/usr/bin/env bash
# Local Kubernetes + HPA autoscaling demo for the RAG API.
#
# Spins up a kind cluster, deploys the API (mock mode, key-free), installs metrics-server,
# drives load with the existing Locust test, and samples HPA + pod count through a full
# scale-up (2 -> N) and scale-down (N -> 2). Everything local and free.
#
# Usage:   make k8s-demo        (or: bash scripts/k8s_demo.sh)
# Tunables (env): LOAD_SECS (default 150), COOLDOWN_SECS (180), USERS (500), CLUSTER (rag-demo)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CLUSTER="${CLUSTER:-rag-demo}"
NS="rag-system"
LOAD_SECS="${LOAD_SECS:-150}"
COOLDOWN_SECS="${COOLDOWN_SECS:-180}"
USERS="${USERS:-500}"
RESULTS="$REPO/k8s/results"
SAMPLES="$RESULTS/hpa_samples.log"
LOCUST="$REPO/.venv-eval/bin/locust"
mkdir -p "$RESULTS"
: > "$SAMPLES"

log() { printf '\n=== %s ===\n' "$1"; }
ready_pods() { kubectl get deploy rag-api -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0; }
hpa_line()  { kubectl get hpa rag-api-hpa -n "$NS" --no-headers 2>/dev/null; }
sample() {  # $1 = phase label
    printf '%s | %-8s | ready_pods=%s | hpa: %s\n' \
        "$(date +%H:%M:%S)" "$1" "$(ready_pods)" "$(hpa_line)" | tee -a "$SAMPLES"
}

# --- 1. kind cluster (fresh) ------------------------------------------------
log "kind cluster '$CLUSTER'"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    echo "deleting existing cluster for a clean run..."
    kind delete cluster --name "$CLUSTER"
fi
kind create cluster --name "$CLUSTER" --config k8s/kind/kind-cluster.yaml || { echo "kind create failed"; exit 1; }
kubectl config use-context "kind-$CLUSTER"

# --- 2. metrics-server (kind needs --kubelet-insecure-tls) ------------------
log "metrics-server"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deployment metrics-server --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl -n kube-system rollout status deployment metrics-server --timeout=150s

# --- 3. build + load the API image ------------------------------------------
log "build + load image rag-system:latest"
docker build -t rag-system:latest . || { echo "docker build failed"; exit 1; }
kind load docker-image rag-system:latest --name "$CLUSTER"

# --- 4. deploy (mock-mode configmap + NodePort; prod configmap/secret skipped) ---
log "deploy"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/kind/config-demo.yaml      # USE_MOCK=true (key-free)
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml           # ClusterIP (prod-style)
kubectl apply -f k8s/kind/api-nodeport.yaml     # NodePort for host load
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment rag-api -n "$NS" --timeout=180s || { kubectl get pods -n "$NS"; exit 1; }

# --- 5. wait for HPA to read CPU metrics (not <unknown>) --------------------
log "waiting for HPA metrics"
for i in $(seq 1 30); do
    line="$(hpa_line)"
    echo "  $line"
    case "$line" in *"cpu: <unknown>"*|"") sleep 6 ;; *) echo "metrics ready."; break ;; esac
done
# sanity: API reachable through the NodePort -> Service -> pods
curl -fsS --retry 10 --retry-delay 2 --retry-connrefused http://localhost:8000/health >/dev/null \
    && echo "API reachable at localhost:8000" || echo "WARN: /health not reachable"

# --- 6. baseline, then drive load -------------------------------------------
log "baseline (pre-load)"; sample baseline; sample baseline
log "load: $USERS users for ${LOAD_SECS}s (Locust via NodePort)"
"$LOCUST" -f tests/load/locustfile.py --host http://localhost:8000 \
    --headless -u "$USERS" -r 50 -t "${LOAD_SECS}s" --csv "$RESULTS/k8s_locust" \
    > "$RESULTS/k8s_locust_run.log" 2>&1 &
LOCUST_PID=$!

end=$((SECONDS + LOAD_SECS))
while [ $SECONDS -lt $end ]; do sample load; sleep 8; done
kill "$LOCUST_PID" 2>/dev/null; wait "$LOCUST_PID" 2>/dev/null

# --- 7. cooldown: observe scale-down ----------------------------------------
log "cooldown ${COOLDOWN_SECS}s (observe scale-down)"
end=$((SECONDS + COOLDOWN_SECS))
while [ $SECONDS -lt $end ]; do sample cooldown; sleep 10; done

# --- 8. summary -------------------------------------------------------------
log "DONE — peak/min ready_pods over the run"
peak=$(grep -oE 'ready_pods=[0-9]+' "$SAMPLES" | grep -oE '[0-9]+' | sort -n | tail -1)
base=$(grep -oE 'ready_pods=[0-9]+' "$SAMPLES" | grep -oE '[0-9]+' | sort -n | head -1)
echo "min ready_pods=$base  peak ready_pods=$peak"
echo "raw samples: $SAMPLES"
echo "locust csv:  $RESULTS/k8s_locust_stats.csv"
