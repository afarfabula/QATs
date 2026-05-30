#!/usr/bin/env bash
# Batch-transfer imagenet-1k parquet files from local /tmp to GPU worker /tmp,
# using the shared system disk (/mlx_devbox) as a staging area.
#
# Pipeline per batch:
#   local /tmp/imagenet1k_full_parquet/data/*.parquet  (source, on master)
#     -- cp -->  $RELAY_DIR (shared, /mlx_devbox/...)
#     -- mlx worker login + cp -->  worker /tmp/imagenet1k_full_parquet/data/
#     -- rm -->  cleanup $RELAY_DIR
#
# Usage:
#   bash transfer_imnet_to_worker.sh meta            # one-off: README/classes.py/.gitattributes
#   bash transfer_imnet_to_worker.sh 0               # batch 0
#   bash transfer_imnet_to_worker.sh 1               # batch 1
#   bash transfer_imnet_to_worker.sh 2               # batch 2
#   bash transfer_imnet_to_worker.sh 3               # batch 3
#   bash transfer_imnet_to_worker.sh all             # meta + all batches sequentially
#   bash transfer_imnet_to_worker.sh status          # show batch plan + worker dst counts
#
# Override defaults via env: WORKER_ID, SRC_DIR, RELAY_DIR, DST_DIR, BATCHES.

set -euo pipefail

WORKER_ID="${WORKER_ID:-928201}"
SRC_DIR="${SRC_DIR:-/tmp/imagenet1k_full_parquet}"
RELAY_DIR="${RELAY_DIR:-/mlx_devbox/users/quyanyi/imnet1k_relay}"
DST_DIR="${DST_DIR:-/tmp/imagenet1k_full_parquet}"
BATCHES="${BATCHES:-4}"

ARG="${1:-}"
if [[ -z "$ARG" ]]; then
  echo "usage: $0 <0..$((BATCHES-1)) | meta | all | status>"
  exit 1
fi

if [[ ! -d "$SRC_DIR/data" ]]; then
  echo "missing src: $SRC_DIR/data"
  exit 2
fi

# Sorted parquet list (deterministic batch slicing)
mapfile -t ALL < <(cd "$SRC_DIR/data" && ls -1 *.parquet | sort)
TOTAL="${#ALL[@]}"
PER=$(( (TOTAL + BATCHES - 1) / BATCHES ))

slice_range() {
  local idx="$1"
  local start=$(( idx * PER ))
  local end=$(( start + PER ))
  (( end > TOTAL )) && end="$TOTAL"
  echo "$start $end"
}

print_status() {
  local i s e n first last_idx last
  echo "src_dir=$SRC_DIR"
  echo "relay_dir=$RELAY_DIR"
  echo "dst_dir=$DST_DIR  (worker $WORKER_ID)"
  echo "total_parquet=$TOTAL  batches=$BATCHES  per=$PER"
  for ((i=0; i<BATCHES; i++)); do
    read s e < <(slice_range "$i")
    n=$(( e - s ))
    first="${ALL[$s]:-<none>}"
    last_idx=$(( e - 1 ))
    last="${ALL[$last_idx]:-<none>}"
    echo "  batch $i: idx[$s,$e) count=$n  first=$first  last=$last"
  done
  echo "--- relay current ---"
  ls -1 "$RELAY_DIR/data" 2>/dev/null | wc -l | awk '{print "relay_data_files="$1}'
  du -sh "$RELAY_DIR" 2>/dev/null | awk '{print "relay_size="$1}'
  echo "--- worker dst current ---"
  mlx worker login "$WORKER_ID" -- "ls -1 $DST_DIR/data 2>/dev/null | wc -l && du -sh $DST_DIR 2>/dev/null" 2>&1 | tail -3
}

copy_local_to_relay() {
  local idx="$1" s e i f src dst src_sz dst_sz
  read s e < <(slice_range "$idx")
  mkdir -p "$RELAY_DIR/data"
  echo "[batch $idx] copy local -> relay  ($((e-s)) files)"
  for ((i=s; i<e; i++)); do
    f="${ALL[$i]}"
    src="$SRC_DIR/data/$f"
    dst="$RELAY_DIR/data/$f"
    src_sz="$(stat -c%s "$src")"
    dst_sz="$(stat -c%s "$dst" 2>/dev/null || echo 0)"
    if [[ "$src_sz" != "$dst_sz" ]]; then
      cp -f "$src" "$dst"
    fi
  done
  du -sh "$RELAY_DIR/data" | awk '{print "relay_after_copy="$1}'
}

worker_pull_from_relay() {
  local idx="$1"
  read s e < <(slice_range "$idx")
  echo "[batch $idx] worker pulls relay -> $DST_DIR/data"
  # The worker can read $RELAY_DIR (shared via /mlx_devbox) and copy to its own /tmp.
  # Keep remote command on a single line to play nice with `mlx worker login -- ...`.
  # The $(...) for counting must be escaped so it runs on the worker, not master.
  mlx worker login "$WORKER_ID" -- "mkdir -p $DST_DIR/data && cp -f $RELAY_DIR/data/*.parquet $DST_DIR/data/ && echo dst_data_files=\"\$(ls -1 $DST_DIR/data | wc -l)\" && df -h $DST_DIR | tail -1"
}

cleanup_relay_batch() {
  local idx="$1" s e i
  read s e < <(slice_range "$idx")
  echo "[batch $idx] cleanup relay"
  for ((i=s; i<e; i++)); do
    rm -f "$RELAY_DIR/data/${ALL[$i]}"
  done
  du -sh "$RELAY_DIR" 2>/dev/null | awk '{print "relay_after_clean="$1}'
}

run_batch() {
  local idx="$1"
  if (( idx < 0 || idx >= BATCHES )); then
    echo "bad batch index: $idx (must be 0..$((BATCHES-1)))"
    exit 3
  fi
  copy_local_to_relay "$idx"
  worker_pull_from_relay "$idx"
  cleanup_relay_batch "$idx"
  echo "[batch $idx] DONE"
}

run_meta() {
  echo "[meta] copy README/classes.py/.gitattributes via relay"
  mkdir -p "$RELAY_DIR"
  cp -f "$SRC_DIR/classes.py"     "$RELAY_DIR/classes.py"
  cp -f "$SRC_DIR/README.md"      "$RELAY_DIR/README.md"
  cp -f "$SRC_DIR/.gitattributes" "$RELAY_DIR/.gitattributes"
  mlx worker login "$WORKER_ID" -- "mkdir -p $DST_DIR && cp -f $RELAY_DIR/classes.py $RELAY_DIR/README.md $RELAY_DIR/.gitattributes $DST_DIR/ && ls -la $DST_DIR | head -10"
  rm -f "$RELAY_DIR/classes.py" "$RELAY_DIR/README.md" "$RELAY_DIR/.gitattributes"
  echo "[meta] DONE"
}

case "$ARG" in
  status) print_status ;;
  meta)   run_meta ;;
  all)
    run_meta
    for ((i=0; i<BATCHES; i++)); do run_batch "$i"; done
    print_status
    ;;
  ''|*[!0-9]*)
    echo "bad arg: $ARG"
    exit 4
    ;;
  *) run_batch "$ARG" ;;
esac
