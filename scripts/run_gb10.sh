#!/usr/bin/env bash
# Every measurement and training run for Phase 3, in order, on the GB10.
# Results land in results/ (JSON), logs in results/logs/. A preflight check
# runs first; steps whose weights or data are missing are skipped with a
# note, and a failed step prints its error, so this can be rerun as things
# arrive.
#
#   bash scripts/gb10_setup.sh                      # once
#   python scripts/fetch_datasets.py all            # MOT17 sequences + CrowdHuman check
#   bash scripts/run_gb10.sh [measure|train|all]    # default: all
#
# Needs, besides the code: weights/yolo26strained.pt and weights/yoloheadv26s.pt
# (not in git — copy them from the laptop), dataset/MOT17 and dataset/CrowdHuman,
# and for the speed benchmarks some videos in videos/.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ ! -f venv/bin/activate ]; then
  echo "No venv here — run: bash scripts/gb10_setup.sh"; exit 1
fi
. venv/bin/activate
export PYTHONIOENCODING=utf-8
WHAT=${1:-all}
LOG=results/logs
mkdir -p "$LOG"

echo "=== preflight"
python scripts/preflight.py --env "$LOG/preflight.env" | tee "$LOG/preflight.log"
. "$LOG/preflight.env"
if [ "$CODE_OK" != 1 ]; then echo "The code doesn't import — fix that first (see above)."; exit 1; fi
DEV=(--device "$DEVICE")

run() {   # run <name> <command...>
  local name=$1; shift
  echo "=== $name: $*"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "    ok ($LOG/$name.log)"
  else
    echo "    FAILED — last lines of $LOG/$name.log:"
    tail -n 15 "$LOG/$name.log" | sed 's/^/      /'
  fi
}
weights() { [ "$WEIGHTS_OK" = 1 ] || { echo "--- skip $1: detector weights missing (see preflight)"; return 1; }; }

CH=(--data crowdhuman:dataset/CrowdHuman)

if [ "$WHAT" = measure ] || [ "$WHAT" = all ]; then
  # 1. Speed: the ≥ 10 FPS / < 2 s targets, normal and dense scenes
  for v in demo.mp4 kumbhvideo4.mp4; do
    if [[ " $VIDEOS " == *" $v "* ]]; then
      weights "bench $v" && run "bench_${v%.mp4}" python evaluate.py bench --source "videos/$v" --frames 300 "${DEV[@]}"
    else
      echo "--- skip bench $v: videos/$v not found"
    fi
  done

  # 2. Tracking and counting on MOT17
  for s in $MOT17_SEQS; do
    weights "mot17 $s" && run "mot17_$s" python evaluate.py mot17 --seq "dataset/MOT17/train/$s" "${DEV[@]}"
  done
  [ -z "$MOT17_SEQS" ] && echo "--- skip mot17: no sequences (python scripts/fetch_datasets.py mot17)"

  # 3. The 20 s forecast against what really happened (MOT17 ground truth; no weights needed)
  if [ -n "$MOT17_SEQS" ]; then
    FC=()
    for s in $MOT17_SEQS; do FC+=("dataset/MOT17/train/$s"); done
    run forecast python evaluate.py forecast --tracks "${FC[@]}" --every 2 --warmup 5
  fi

  # 4. Crowd counting on CrowdHuman val (detectors; the point model if already trained)
  if [ "$CROWDHUMAN_OK" = 1 ]; then
    weights "dense count" && run dense_detectors python evaluate.py dense "${CH[@]}" --max 500 "${DEV[@]}"
  else
    echo "--- skip dense count: dataset/CrowdHuman not found"
  fi
fi

if [ "$WHAT" = train ] || [ "$WHAT" = all ]; then
  # 5. Dense-crowd point model on CrowdHuman heads (+ your own labelled frames, if any)
  if [ "$CROWDHUMAN_OK" = 1 ]; then
    TRAIN=("${CH[@]}")
    [ "$KUMBH_POINTS_OK" = 1 ] && TRAIN+=(--data points:dataset/kumbh_points)
    AMP=(); [ "$DEVICE" != cpu ] && AMP=(--amp)
    run train_dense python train_dense.py "${TRAIN[@]}" --init imagenet --epochs 100 "${AMP[@]}" \
        --batch 8 --val-max 300 --device "$DEVICE" --out weights/p2pnet_crowd.pth
    if [ -f weights/p2pnet_crowd.pth ]; then
      weights "dense count (trained)" && run dense_trained python evaluate.py dense "${CH[@]}" --max 500 \
          --weights weights/p2pnet_crowd.pth "${DEV[@]}"
      [[ " $VIDEOS " == *" kumbhvideo4.mp4 "* ]] && weights "bench dense" && \
        run bench_dense_points python evaluate.py bench --source videos/kumbhvideo4.mp4 --frames 300 "${DEV[@]}"
    fi
  else
    echo "--- skip training: dataset/CrowdHuman not found (see scripts/fetch_datasets.py)"
  fi
fi
echo "Done. JSON results: results/eval_*.json · logs: $LOG/"
