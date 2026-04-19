#!/bin/bash
# One-shot: migrate completed dataset tars from BN→HDFS, delete BN copies
set -uo pipefail

BN_OUT="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output"
HDFS_BASE="/mnt/hdfs/weichow/maskedit/t2i-pt"

log() { echo "$(date '+%H:%M:%S') [CLEANUP] $*"; }

# Completed datasets with tars still on BN
for DS_DIR in "$BN_OUT"/*/; do
    DS=$(basename "$DS_DIR")
    TAR_COUNT=$(ls "$DS_DIR"/*.tar 2>/dev/null | wc -l)
    JSON_COUNT=$(ls "$DS_DIR"/*.json 2>/dev/null | wc -l)

    if [ "$TAR_COUNT" -eq 0 ] && [ "$JSON_COUNT" -eq 0 ]; then
        continue
    fi

    log "=== $DS: $TAR_COUNT tars, $JSON_COUNT jsons ==="

    # Create HDFS tar directory
    HDFS_TAR_DIR="$HDFS_BASE/${DS}_tars"
    mkdir -p "$HDFS_TAR_DIR" 2>/dev/null || true

    # Migrate tars
    FREED=0
    MIGRATED=0
    for TAR in "$DS_DIR"/*.tar; do
        [ -f "$TAR" ] || continue
        FNAME=$(basename "$TAR")
        DEST="$HDFS_TAR_DIR/$FNAME"

        if [ -f "$DEST" ]; then
            SRC_SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
            DST_SZ=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
            if [ "$SRC_SZ" = "$DST_SZ" ]; then
                rm -f "$TAR"
                FREED=$((FREED + SRC_SZ))
                MIGRATED=$((MIGRATED + 1))
                continue
            fi
        fi

        SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
        cat "$TAR" > "$DEST" 2>/dev/null
        DST_SZ=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        if [ "$DST_SZ" = "$SZ" ]; then
            rm -f "$TAR"
            FREED=$((FREED + SZ))
            MIGRATED=$((MIGRATED + 1))
        else
            log "  WARN: size mismatch $FNAME ($SZ vs $DST_SZ)"
        fi

        if [ $((MIGRATED % 200)) -eq 0 ] && [ "$MIGRATED" -gt 0 ]; then
            log "  $DS: $MIGRATED tars migrated, $(echo "scale=1; $FREED/1073741824" | bc)GB freed"
        fi
    done

    # Delete BN JSONs (already on HDFS via upload_to_hdfs)
    rm -f "$DS_DIR"/*.json 2>/dev/null

    # Try remove empty dir
    rmdir "$DS_DIR" 2>/dev/null || true

    log "$DS done: $MIGRATED tars migrated, $(echo "scale=1; $FREED/1073741824" | bc)GB freed"
done

log "=== All cleanup done ==="
