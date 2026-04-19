#!/bin/bash
# Migrate existing BN tars to HDFS (via FUSE mount) and delete BN copies
# Run on any machine with access to /mnt/bn/ and /mnt/hdfs/
set -uo pipefail

BN_DATA="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/data"
HDFS_T2I="/mnt/hdfs/weichow/maskedit/t2i"
LOGDIR="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs"
mkdir -p "$LOGDIR"

log() { echo "$(date '+%H:%M:%S') [MIGRATE] $*"; }

# Files to migrate
declare -a SRCS=(
    "$BN_DATA/textatlas5m/textatlas5m_images.tar"
    "$BN_DATA/mico150k/mico150k_images.tar"
    "$BN_DATA/pico_banana/pico_banana_images.tar"
)

mkdir -p "$HDFS_T2I" 2>/dev/null || true

for SRC in "${SRCS[@]}"; do
    FNAME=$(basename "$SRC")
    DEST="$HDFS_T2I/$FNAME"

    if [ ! -f "$SRC" ]; then
        log "SKIP: $SRC not found"
        continue
    fi

    SRC_SIZE=$(stat -c%s "$SRC" 2>/dev/null || stat -f%z "$SRC" 2>/dev/null || echo 0)
    SRC_GB=$(python3 -c "print(f'{${SRC_SIZE}/1073741824:.1f}')")
    log "START: $FNAME (${SRC_GB}GB) → $DEST"

    # Check if already on HDFS
    if [ -f "$DEST" ]; then
        DEST_SIZE=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST" 2>/dev/null || echo 0)
        if [ "$DEST_SIZE" = "$SRC_SIZE" ]; then
            log "ALREADY ON HDFS (same size: $SRC_SIZE), deleting BN copy..."
            rm -f "$SRC"
            log "DELETED: $SRC"
            continue
        else
            log "HDFS size mismatch ($DEST_SIZE vs $SRC_SIZE), re-copying..."
        fi
    fi

    # Copy via FUSE (cat for large files to avoid FUSE cp issues)
    log "Copying via cat pipe (large file)..."
    if cat "$SRC" > "$DEST"; then
        DEST_SIZE=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST" 2>/dev/null || echo 0)
        if [ "$DEST_SIZE" = "$SRC_SIZE" ]; then
            log "VERIFIED: size matches ($SRC_SIZE bytes)"
            rm -f "$SRC"
            log "DELETED BN copy: $SRC"
        else
            log "WARNING: Size mismatch! BN=$SRC_SIZE HDFS=$DEST_SIZE. Keeping BN."
        fi
    else
        log "FAILED: cat copy for $FNAME"
    fi
done

log "Migration complete!"
