#!/usr/bin/env bash
set -euo pipefail

steps="${TRACK3_TRAIN_STEPS:-${SCREEN_STEPS:-500}}"
trials="${TRACK3_NUM_TRIALS:-1}"
source_script="${TRACK3_SOURCE:-records/track_3_optimization/train_gpt_simple.py}"
generated_script="${TRACK3_GENERATED_SCRIPT:-/tmp/train_gpt_simple_locoprop_m_${steps}.py}"
nproc="${NPROC_PER_NODE:-1}"

python3 tools/make_track3_locoprop_m.py \
  --source "${source_script}" \
  --output "${generated_script}" \
  --steps "${steps}"

python3 -m py_compile "${generated_script}"

source_sha="$(sha256sum "${source_script}" | awk '{print $1}')"
generator_sha="$(sha256sum tools/make_track3_locoprop_m.py | awk '{print $1}')"
generated_sha="$(sha256sum "${generated_script}" | awk '{print $1}')"

trial_arg_mode="${TRACK3_TRIAL_ARG_MODE:-}"
if [[ -z "${trial_arg_mode}" ]]; then
  if [[ "${TRACK3_PASS_TRIAL_ARG:-1}" == "1" ]]; then
    trial_arg_mode="$(python3 - "${source_script}" <<'PY'
from pathlib import Path
import sys
code = Path(sys.argv[1]).read_text().split("\n====================================================================================================", 1)[0]
print("optional-seed" if 'add_argument("--seed"' in code else "positional-count")
PY
)"
  else
    trial_arg_mode="none"
  fi
fi

echo "track3_locom_runner source=${source_script} generated=${generated_script} steps=${steps} trials=${trials} nproc=${nproc} mbs=${TRACK3_MBS:-64} trial_arg_mode=${trial_arg_mode}"
echo "track3_locom_source_sha source=${source_sha} generator=${generator_sha} generated=${generated_sha}"

if [[ "${TRACK3_DRY_RUN:-0}" == "1" ]]; then
  echo "generated ${generated_script}"
  exit 0
fi

case "${trial_arg_mode}" in
  positional-count)
    torchrun --standalone --nproc_per_node="${nproc}" "${generated_script}" "${trials}"
    ;;
  none)
    torchrun --standalone --nproc_per_node="${nproc}" "${generated_script}"
    ;;
  optional-seed)
    for ((trial = 0; trial < trials; trial++)); do
      seed=$(( ${TRACK3_SEED_BASE:-0} + ${TRACK3_SEED_OFFSET:-0} + trial ))
      TRACK3_SEED_OFFSET=$(( ${TRACK3_SEED_OFFSET:-0} + trial )) \
        torchrun --standalone --nproc_per_node="${nproc}" "${generated_script}" --seed "${seed}"
    done
    ;;
  *)
    echo "Unsupported TRACK3_TRIAL_ARG_MODE=${trial_arg_mode}; use positional-count, none, or optional-seed" >&2
    exit 2
    ;;
esac
