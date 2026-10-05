#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/root/wr-fresh-20260526}"
label="${RUN_LABEL:-track3_locom_nm}"
steps="${TRACK3_TRAIN_STEPS:-3300}"
trials="${TRACK3_NUM_TRIALS:-2}"
nproc="${NPROC_PER_NODE:-1}"
data_chunks="${TRACK3_DATA_CHUNKS:-20}"
target_loss="${TRACK3_TARGET_LOSS:-3.28}"
source_script="${TRACK3_SOURCE:-records/track_3_optimization/results/20260505_newton_muon/train_gpt_simple_newton_muon.py}"
torch_version="${TRACK3_TORCH_VERSION:-2.10.0}"
torch_index_url="${TRACK3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor-track3-locom-nm-${nproc}x}"

mkdir -p /root/prime_track3_logs
cd "${repo_dir}"

if [[ ! -x /root/venv/bin/python ]]; then
  apt-get update
  apt-get install -y --no-install-recommends python3 python3-venv python3-pip git ca-certificates curl
  python3 -m venv /root/venv
fi

source /root/venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q --index-url "${torch_index_url}" "torch==${torch_version}"
python -m pip install -q numpy tqdm huggingface-hub typing-extensions setuptools

python data/cached_fineweb10B.py "${data_chunks}" > "/root/prime_track3_logs/${label}_cache_fineweb_${data_chunks}.log" 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "/root/prime_track3_logs/${label}_gpu.txt" 2>&1 || true

generated="/root/prime_track3_logs/${label}_train_gpt_simple_locom.py"
TRACK3_SOURCE="${source_script}" \
TRACK3_GENERATED_SCRIPT="${generated}" \
TRACK3_TRAIN_STEPS="${steps}" \
TRACK3_DRY_RUN=1 \
tools/run_track3_locoprop_m.sh > "/root/prime_track3_logs/${label}_generate.log" 2>&1

run_log="/root/prime_track3_logs/${label}.log"
status_file="/root/prime_track3_logs/${label}.status"
{
  printf 'started %s\n' "$(date -Is)"
  printf 'repo=%s commit=%s\n' "${repo_dir}" "$(git rev-parse --short HEAD 2>/dev/null || true)"
  printf 'label=%s nproc=%s steps=%s trials=%s target=%s data_chunks=%s source=%s torch=%s torch_index=%s\n' \
    "${label}" "${nproc}" "${steps}" "${trials}" "${target_loss}" "${data_chunks}" "${source_script}" "${torch_version}" "${torch_index_url}"
  printf 'locom layers=%s K=%s sample_tokens=%s inner_lr=%s prox=%s alpha=%s norm_cap=%s mbs=%s gather=%s accum=%s\n' \
    "${TRACK3_LOCOM_LAYERS:-all}" "${TRACK3_LOCOM_STEPS:-4}" "${TRACK3_LOCOM_SAMPLE_TOKENS:-1024}" \
    "${TRACK3_LOCOM_INNER_LR:-0.1}" "${TRACK3_LOCOM_PROX:-0.1}" "${TRACK3_LOCOM_ALPHA:-1.0}" \
    "${TRACK3_LOCOM_NORM_CAP:-0.20}" "${TRACK3_MBS:-64}" "${TRACK3_LOCOM_GATHER_SAMPLES:-1}" \
    "${TRACK3_LOCOM_ACCUM_SAMPLES:-1}"
} | tee "${status_file}"

rc=0
env \
  TRACK3_SOURCE="${source_script}" \
  TRACK3_GENERATED_SCRIPT="${generated}" \
  TRACK3_TRAIN_STEPS="${steps}" \
  TRACK3_NUM_TRIALS="${trials}" \
  TRACK3_TARGET_LOSS="${target_loss}" \
  TRACK3_MBS="${TRACK3_MBS:-64}" \
  TRACK3_LOCOM_LAYERS="${TRACK3_LOCOM_LAYERS:-all}" \
  TRACK3_LOCOM_STEPS="${TRACK3_LOCOM_STEPS:-4}" \
  TRACK3_LOCOM_SAMPLE_TOKENS="${TRACK3_LOCOM_SAMPLE_TOKENS:-1024}" \
  TRACK3_LOCOM_INNER_LR="${TRACK3_LOCOM_INNER_LR:-0.1}" \
  TRACK3_LOCOM_TARGET_GAMMA="${TRACK3_LOCOM_TARGET_GAMMA:-1.0}" \
  TRACK3_LOCOM_PROX="${TRACK3_LOCOM_PROX:-0.1}" \
  TRACK3_LOCOM_ALPHA="${TRACK3_LOCOM_ALPHA:-1.0}" \
  TRACK3_LOCOM_NORM_CAP="${TRACK3_LOCOM_NORM_CAP:-0.20}" \
  TRACK3_LOCOM_NORM_TO_BASE="${TRACK3_LOCOM_NORM_TO_BASE:-0}" \
  TRACK3_LOCOM_GATHER_SAMPLES="${TRACK3_LOCOM_GATHER_SAMPLES:-1}" \
  TRACK3_LOCOM_ACCUM_SAMPLES="${TRACK3_LOCOM_ACCUM_SAMPLES:-1}" \
  TRACK3_LOCOM_MICRO_SAMPLE_TOKENS="${TRACK3_LOCOM_MICRO_SAMPLE_TOKENS:-0}" \
  TRACK3_SEED_BASE="${TRACK3_SEED_BASE:-0}" \
  TRACK3_SEED_OFFSET="${TRACK3_SEED_OFFSET:-0}" \
  SCREEN_VAL_EVERY="${SCREEN_VAL_EVERY:-125}" \
  NPROC_PER_NODE="${nproc}" \
  bash tools/run_track3_locoprop_m.sh > "${run_log}" 2>&1 || rc=$?

printf 'finished %s rc=%s\n' "$(date -Is)" "${rc}" | tee -a "${status_file}"
touch "/root/prime_track3_logs/${label}.DONE"
exit "${rc}"
