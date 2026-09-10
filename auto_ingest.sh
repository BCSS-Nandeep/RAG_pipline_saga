#!/bin/bash
# Auto-incremental RAG ingestion — runs every hour via cron
# Only processes new documents since the last run

PIPELINE_DIR="/Users/bhaskarlekkala/BluraSaga/rag_pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG_FILE="$PIPELINE_DIR/auto_ingest.log"

cd "$PIPELINE_DIR" || exit 1

echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting incremental ingestion" >> "$LOG_FILE"

$PYTHON pipeline.py --ingest >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished with exit code $EXIT_CODE" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep log file from growing too large (keep last 5000 lines)
tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
