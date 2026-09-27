#!/usr/bin/env bash
# Ablation study for STAMP on MATE-4v8-9: remove one component at a time.
#
#   scripts/run_ablation.sh tmux       # launch everything in tmux session "ablation"
#   scripts/run_ablation.sh status     # which runs are done
#   scripts/run_ablation.sh eval       # score every variant whose runs are finished
#   scripts/run_ablation.sh report     # table -> runs/ablation/<FULL_TAG>/ABLATION.md
#
# Every variant, the full pipeline included, is trained here under the same step
# budget and seeds, so the comparison is budget-matched:
#   train-time  full, no_firsthand, no_age, no_consensus -> runs/abl_<v>/seed<N>
#   eval-time   no_flow, no_soft, no_intent, no_search, no_voronoi, scored on the
#               abl_full checkpoints (the component lives in the planner, not in
#               any weights), so each is paired with the full row.
# Every row replays the same episode seeds and runs on the same base environment:
# the camera channel is always on, through MATE's RestrictedCommunicationRange at
# env.comm_range, so it is not an ablation variant.  Training logs to wandb,
# project $WANDB_PROJECT, one group per variant.
#
# Knobs (environment): SEEDS, STEPS, EPISODES, EVAL_SEED, JOBS, THREADS,
#                      FULL_TAG, OUT, WANDB_PROJECT, SESSION
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/miniconda3/envs/mate/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-STAMP}"

SEEDS="${SEEDS:-1 2 3 4 5}"
STEPS="${STEPS:-50000}"            # the flow head converges well before this
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-12345}"
JOBS="${JOBS:-48}"                 # parallel eval processes
THREADS="${THREADS:-4}"            # torch/OMP threads per process
FULL_TAG="${FULL_TAG:-abl_full}"   # checkpoints the eval-time variants are scored on
SESSION="${SESSION:-ablation}"
OUT="${OUT:-runs/ablation/${FULL_TAG}}"   # eval rows and ABLATION.md

TRAIN_VARIANTS="full no_firsthand no_age no_consensus"
EVAL_VARIANTS="no_flow no_soft no_intent no_search no_voronoi"

train_overrides() {
  case "$1" in
    full)         echo '' ;;
    no_firsthand) echo '--set=env.belief_drop=["first_hand"]' ;;
    no_age)       echo '--set=env.belief_drop=["age"]' ;;
    no_consensus) echo '--set=trajectory.consensus_weight=0.0' ;;
    *) echo "unknown train variant $1" >&2; exit 1 ;;
  esac
}

threads() { export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"; }

all_trained() {
  for v in $TRAIN_VARIANTS; do
    for s in $SEEDS; do
      [[ -f "runs/abl_${v}/seed${s}/results.json" ]] || return 1
    done
  done
}

# One training run in the foreground, so a tmux window shows it live.
cmd_train_one() {
  local v="$1" s="$2" dir="runs/abl_$1/seed$2"
  threads
  mkdir -p logs
  if [[ -f "$dir/results.json" ]]; then echo "$dir already done"; return 0; fi
  rm -rf "$dir"   # no resume: a half run is restarted from zero
  # shellcheck disable=SC2046
  python train.py --seed "$s" --tag "abl_${v}" --steps "$STEPS" \
    --device cpu --final-episodes 50 --wandb $(train_overrides "$v") \
    2>&1 | tee "logs/abl_${v}_seed${s}.log"
}

cmd_tmux() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
  fi
  local repo; repo="$(pwd)"
  local knobs="SEEDS='$SEEDS' STEPS=$STEPS EPISODES=$EPISODES EVAL_SEED=$EVAL_SEED JOBS=$JOBS THREADS=$THREADS FULL_TAG=$FULL_TAG OUT=$OUT WANDB_PROJECT=$WANDB_PROJECT"
  local first=1
  for v in $TRAIN_VARIANTS; do
    for s in $SEEDS; do
      local name="${v}-s${s}"
      if (( first )); then
        tmux new-session -d -s "$SESSION" -n "$name" -c "$repo"; first=0
      else
        tmux new-window -t "$SESSION" -n "$name" -c "$repo"
      fi
      # send-keys into a shell rather than a window command, so the window
      # stays open with the final lines on screen after the run exits.
      tmux send-keys -t "$SESSION:$name" \
        "$knobs scripts/run_ablation.sh train-one $v $s; echo EXIT=\$?" C-m
    done
  done
  tmux new-window -t "$SESSION" -n eval -c "$repo"
  tmux send-keys -t "$SESSION:eval" "$knobs scripts/run_ablation.sh wait-eval; echo EXIT=\$?" C-m
  echo "tmux session '$SESSION': $(tmux list-windows -t "$SESSION" | wc -l) windows"
  echo "attach: tmux attach -t $SESSION   (Ctrl-b w lists windows, Ctrl-b d detaches)"
}

cmd_wait_eval() {
  until all_trained; do
    clear; date; cmd_status; sleep 300
  done
  cmd_status
  cmd_eval
  cmd_report
}

cmd_eval() {
  threads
  local jobs; jobs=$(mktemp)
  for s in $SEEDS; do
    # results.json is written last, so its presence means the weights are final.
    [[ -f "runs/${FULL_TAG}/seed${s}/results.json" ]] || continue
    for v in $EVAL_VARIANTS; do
      out="$OUT/eval/$v/seed${s}.json"
      [[ -f "$out" ]] || echo "$v runs/${FULL_TAG}/seed${s}/checkpoint.pt $out" >> "$jobs"
    done
  done
  for v in $TRAIN_VARIANTS; do
    for s in $SEEDS; do
      [[ -f "runs/abl_${v}/seed${s}/results.json" ]] || continue
      out="$OUT/eval/$v/seed${s}.json"
      [[ -f "$out" ]] || echo "$v runs/abl_${v}/seed${s}/checkpoint.pt $out" >> "$jobs"
    done
  done
  echo "$(wc -l < "$jobs") eval jobs, $JOBS in parallel, $EPISODES episodes each"
  mkdir -p "$OUT/logs"
  xargs -P "$JOBS" -L 1 bash -c '
    python ablation.py eval --variant "$0" --checkpoint "$1" --out "$2" \
      --episodes '"$EPISODES"' --seed '"$EVAL_SEED"' \
      > "'"$OUT"'/logs/$0_$(basename "$2" .json).log" 2>&1 \
      && echo "done $0 $2" || echo "FAILED $0 $2"' < "$jobs"
  rm -f "$jobs"
}

cmd_report() {
  python ablation.py report --root "$OUT/eval" --markdown "$OUT/ABLATION.md"
}

cmd_status() {
  for v in $TRAIN_VARIANTS; do
    for s in $SEEDS; do
      dir="runs/abl_${v}/seed${s}"
      if [[ -f "$dir/results.json" ]]; then state=done
      elif [[ -s "$dir/metrics.jsonl" ]]; then
        state="running $(tail -1 "$dir/metrics.jsonl" | python -c 'import json,sys;print(json.load(sys.stdin)["env_steps"])') / $STEPS"
      else state=missing; fi
      echo "$dir: $state"
    done
  done
  echo "eval rows: $(ls "$OUT"/eval/*/seed*.json 2>/dev/null | wc -l)"
}

case "${1:-}" in
  tmux) cmd_tmux ;;
  train-one) cmd_train_one "$2" "$3" ;;
  wait-eval) cmd_wait_eval ;;
  eval) cmd_eval ;;
  report) cmd_report ;;
  status) cmd_status ;;
  *) sed -n 2,21p "$0"; exit 1 ;;
esac
