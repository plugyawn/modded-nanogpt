#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/root/wr-fresh-20260526}"
label="${RUN_LABEL:-wr_record_resume_probe}"
steps="${WR_TRAIN_STEPS:-3040}"
schedule_steps="${WR_SCHEDULE_STEPS:-3105}"
nproc="${NPROC_PER_NODE:-1}"
seed="${WR_SEED:-28}"
torch_version="${WR_TORCH_VERSION:-2.11.0}"
torch_index_url="${WR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
default_data_chunks=$(( ((steps + 1) * 524288 + 100000000 - 1) / 100000000 ))
data_chunks="${WR_DATA_CHUNKS:-${default_data_chunks}}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor-wr-record-resume-${nproc}x}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

mkdir -p /root/prime_track3_logs
cd "${repo_dir}"

if [[ ! -x /root/venv/bin/python ]]; then
  apt-get update
  apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-dev build-essential git ca-certificates curl
  python3 -m venv /root/venv
fi

source /root/venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q --index-url "${torch_index_url}" "torch==${torch_version}"
python -m pip install -q numpy tqdm huggingface-hub typing-extensions setuptools

python data/cached_fineweb10B.py "${data_chunks}" > "/root/prime_track3_logs/${label}_cache_fineweb_${data_chunks}.log" 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "/root/prime_track3_logs/${label}_gpu.txt" 2>&1 || true

generated="/root/prime_track3_logs/${label}_train_gpt_wr_record_resume.py"
WR_RESUME_DRY_RUN=1 \
WR_RESUME_GENERATED_SCRIPT="${generated}" \
WR_TRAIN_STEPS="${steps}" \
WR_SCHEDULE_STEPS="${schedule_steps}" \
WR_SEED="${seed}" \
NPROC_PER_NODE="${nproc}" \
bash tools/run_wr_record_resume.sh > "/root/prime_track3_logs/${label}_generate.log" 2>&1

run_log="/root/prime_track3_logs/${label}.log"
status_file="/root/prime_track3_logs/${label}.status"
{
  printf 'started %s\n' "$(date -Is)"
  printf 'repo=%s commit=%s\n' "${repo_dir}" "$(git rev-parse --short HEAD 2>/dev/null || true)"
  printf 'label=%s nproc=%s steps=%s schedule_steps=%s seed=%s data_chunks=%s torch=%s torch_index=%s\n' \
    "${label}" "${nproc}" "${steps}" "${schedule_steps}" "${seed}" "${data_chunks}" "${torch_version}" "${torch_index_url}"
  printf 'resume checkpoint=%s model_checkpoint=%s optimizer_checkpoint=%s advance_data=%s restore_rng=%s load_adam=%s load_optimizers=%s target=%s\n' \
    "${WR_RESUME_CHECKPOINT:-}" "${WR_RESUME_MODEL_CHECKPOINT:-}" "${WR_RESUME_OPTIMIZER_CHECKPOINT:-}" \
    "${WR_RESUME_ADVANCE_DATA:-1}" "${WR_RESUME_RESTORE_RNG:-1}" \
    "${WR_RESUME_LOAD_ADAM:-0}" "${WR_RESUME_LOAD_OPTIMIZERS:-0}" "${WR_TARGET_LOSS:-3.28}"
  printf 'save checkpoint=%s save_step=%s\n' "${WR_SAVE_CHECKPOINT:-}" "${WR_SAVE_CHECKPOINT_STEP:-}"
} | tee "${status_file}"

env \
  WR_RESUME_GENERATED_SCRIPT="${generated}" \
  WR_TRAIN_STEPS="${steps}" \
  WR_SCHEDULE_STEPS="${schedule_steps}" \
  WR_SEED="${seed}" \
  NPROC_PER_NODE="${nproc}" \
  WR_RESUME_CHECKPOINT="${WR_RESUME_CHECKPOINT:-}" \
  WR_RESUME_MODEL_CHECKPOINT="${WR_RESUME_MODEL_CHECKPOINT:-}" \
  WR_RESUME_OPTIMIZER_CHECKPOINT="${WR_RESUME_OPTIMIZER_CHECKPOINT:-}" \
  WR_RESUME_ADVANCE_DATA="${WR_RESUME_ADVANCE_DATA:-1}" \
  WR_RESUME_RESTORE_RNG="${WR_RESUME_RESTORE_RNG:-1}" \
  WR_RESUME_LOAD_ADAM="${WR_RESUME_LOAD_ADAM:-0}" \
  WR_RESUME_LOAD_OPTIMIZERS="${WR_RESUME_LOAD_OPTIMIZERS:-0}" \
  WR_RESUME_STRICT_MODEL="${WR_RESUME_STRICT_MODEL:-1}" \
  WR_SAVE_CHECKPOINT="${WR_SAVE_CHECKPOINT:-}" \
  WR_SAVE_CHECKPOINT_STEP="${WR_SAVE_CHECKPOINT_STEP:--1}" \
  WR_TARGET_LOSS="${WR_TARGET_LOSS:-3.28}" \
  SCREEN_VAL_EVERY="${SCREEN_VAL_EVERY:-125}" \
  bash tools/run_wr_record_resume.sh > "${run_log}" 2>&1

rc=$?
printf 'finished %s rc=%s\n' "$(date -Is)" "${rc}" | tee -a "${status_file}"
touch "/root/prime_track3_logs/${label}.DONE"
exit "${rc}"
