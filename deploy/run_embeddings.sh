#!/usr/bin/env bash
# Rebuild the SOC-EYE vector corpus, for as long as it takes.
#
# Why this exists: the vectors already in `test` were written by a different
# embedding model and carry no content_hash, so the API's 8-hourly scheduler
# classifies every one of them as "legacy" and skips it forever. Only a full
# rebuild replaces them with nomic vectors that the scheduler can then keep in
# sync incrementally.
#
# The run is resumable: regenerate_embeddings.py checkpoints per collection in
# MongoDB, so a crash, reboot or OOM loses at most the page in flight. This
# script restarts it until it reports success, which is what makes it safe to
# leave running unattended for hours or days.
#
#   ./run_embeddings.sh                 # every collection in EMBED_COLLECTIONS
#   ./run_embeddings.sh alerts,grievances
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
APP_DIR="$PWD"
PY="${PY:-$APP_DIR/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

COLLECTIONS="${1:-}"
ARGS=()
[ -n "$COLLECTIONS" ] && ARGS+=(--collections "$COLLECTIONS")

LOG_DIR="$APP_DIR/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/embed_$(date +%Y%m%d_%H%M%S).log"
# Set once the retire step has run, so a resume never retires the partially
# rebuilt corpus it is supposed to be continuing.
RETIRED_MARKER="$APP_DIR/.retired_once"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== SOC-EYE embedding rebuild ==="
log "app=$APP_DIR  python=$PY  collections=${COLLECTIONS:-<all from EMBED_COLLECTIONS>}"

log "--- preflight ---"
if ! "$PY" regenerate_embeddings.py --preflight "${ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "PREFLIGHT FAILED — fix the reported problem before rebuilding. Nothing was changed."
    exit 1
fi

attempt=0
while : ; do
    attempt=$((attempt + 1))
    RESUME=()
    if [ -f "$RETIRED_MARKER" ]; then
        RESUME+=(--skip-retire)
        log "--- rebuild attempt $attempt (resuming; old vectors already retired) ---"
    else
        log "--- rebuild attempt $attempt (first pass; retires old vectors by renaming) ---"
    fi

    "$PY" regenerate_embeddings.py --rebuild "${ARGS[@]}" "${RESUME[@]}" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}

    # The first pass renames the old vector collections; from here on every
    # retry must resume rather than retire again.
    touch "$RETIRED_MARKER"

    if [ "$rc" -eq 0 ]; then
        log "rebuild finished cleanly after $attempt attempt(s)"
        break
    fi

    log "rebuild exited rc=$rc — resuming from the last checkpoint in 60s"
    sleep 60
done

log "--- verify ---"
"$PY" regenerate_embeddings.py --verify 2>&1 | tee -a "$LOG"

log "=== done ==="
log "Next: set INGEST_SCHEDULER_ENABLED=true and restart the API so the"
log "8-hourly scheduler keeps the corpus current from here on."
log "Old vectors were renamed, not dropped. Reclaim that space with:"
log "  $PY regenerate_embeddings.py --drop-old"
