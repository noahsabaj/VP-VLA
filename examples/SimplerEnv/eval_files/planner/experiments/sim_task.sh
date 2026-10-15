#!/bin/bash
# One sim run with the paper's SimplerEnv settings, for smoke tests; the servers it uses must already run.
# Usage: sim_task.sh GPU TASK START END OUT [ARGS...]
#   Episodes START..END-1 of TASK. OUT: a run folder in $RUNS_DIR (or an absolute path); it is replaced and gets
#   sim.log. ARGS choose the controller and its servers, e.g.
#   --controller planner --planner mplib --target-source camera --sam3-port 10094
[ $# -ge 5 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; TASK=$2; START=$3; END=$4; OUT=$(run_dir "$5")
shift 5
sim $GPU $TASK $START $END "$OUT" "$@"
