#!/usr/bin/env bash
# Show the talk's Flyte reports in your browser.
#
#   bash show_reports.sh            # comparison report ("Why fine-tune?"), no training
#
# Needs Docker running (on Windows: Docker Desktop with WSL integration on).
# Starts a local Flyte devbox if one is not already running, waits until it is
# ready, then runs the `main` task and prints a link to open.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source .venv/bin/activate

if ! docker ps >/dev/null 2>&1; then
  echo "Docker is not running. Open Docker Desktop (Settings > Resources > WSL integration on), then rerun." >&2
  exit 1
fi

state="$(docker inspect -f '{{.State.Status}}' flyte-devbox 2>/dev/null || echo missing)"
case "$state" in
  running)
    echo "[reports] Flyte devbox already running." ;;
  paused)   # what `flyte stop devbox` leaves behind
    echo "[reports] Resuming the paused Flyte devbox..."
    docker unpause flyte-devbox >/dev/null ;;
  exited|created)
    echo "[reports] Restarting the stopped Flyte devbox..."
    docker start flyte-devbox >/dev/null ;;
  *)
    # A previous devbox can leave a root-owned /tmp/.kube behind, which blocks a
    # new start ("Permission denied removing stale kubeconfig"). Remove it as root via Docker.
    if [ -e /tmp/.kube ]; then
      docker run --rm -v /tmp:/host-tmp alpine rm -rf /host-tmp/.kube
    fi
    echo "[reports] Starting a Flyte devbox (the first time downloads its image)..."
    flyte start devbox ;;
esac

echo "[reports] Waiting for the devbox to be ready..."
for _ in $(seq 1 60); do
  docker exec flyte-devbox kubectl -n flyte get pod -l app.kubernetes.io/name=rustfs >/dev/null 2>&1 && break
  sleep 5
done
docker exec flyte-devbox kubectl -n flyte wait --for=condition=ready pod \
  -l app.kubernetes.io/name=rustfs --timeout=300s >/dev/null

if [ ! -f .flyte/devbox.yaml ]; then
  mkdir -p .flyte
  flyte create config --devbox -o .flyte/devbox.yaml
fi

echo "[reports] Running the comparison report on the devbox..."
flyte -c .flyte/devbox.yaml run --copy-style all workflow.py main

echo
echo "[reports] Open the link above and click the Reports tab (UI: http://localhost:30080/v2)."
echo "[reports] When you are done: flyte stop devbox"
