#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/root/wr-fresh-20260526}"
label="${RUN_LABEL:-wr_record_locom_probe}"
steps="${WR_TRAIN_STEPS:-3040}"
schedule_steps="${WR_SCHEDULE_STEPS:-3105}"
nproc="${NPROC_PER_NODE:-1}"
seed="${WR_SEED:-28}"
torch_version="${WR_TORCH_VERSION:-2.11.0}"
torch_index_url="${WR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
default_data_chunks=$(( ((steps + 1) * 524288 + 100000000 - 1) / 100000000 ))
data_chunks="${WR_DATA_CHUNKS:-${default_data_chunks}}"

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor-wr-record-locom-${nproc}x}"
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

generated="/root/prime_track3_logs/${label}_train_gpt_wr_record_locom.py"
WR_LOCOM_DRY_RUN=1 \
WR_LOCOM_GENERATED_SCRIPT="${generated}" \
WR_TRAIN_STEPS="${steps}" \
WR_SCHEDULE_STEPS="${schedule_steps}" \
WR_SEED="${seed}" \
NPROC_PER_NODE="${nproc}" \
bash tools/run_wr_record_locoprop_m.sh > "/root/prime_track3_logs/${label}_generate.log" 2>&1

run_log="/root/prime_track3_logs/${label}.log"
status_file="/root/prime_track3_logs/${label}.status"
{
  printf 'started %s\n' "$(date -Is)"
  printf 'repo=%s commit=%s\n' "${repo_dir}" "$(git rev-parse --short HEAD 2>/dev/null || true)"
  printf 'label=%s nproc=%s steps=%s schedule_steps=%s seed=%s data_chunks=%s torch=%s torch_index=%s\n' \
    "${label}" "${nproc}" "${steps}" "${schedule_steps}" "${seed}" "${data_chunks}" "${torch_version}" "${torch_index_url}"
  printf 'wr_locom enabled=%s layers=%s windows=%s K=%s sample_tokens=%s inner_lr=%s prox=%s alpha=%s norm_cap=%s norm_to_base=%s min_cos=%s require_loss_decrease=%s\n' \
    "${WR_LOCOM_ENABLED:-1}" "${WR_LOCOM_LAYERS:-all}" "${WR_LOCOM_ACTIVE_WINDOWS:-}" \
    "${WR_LOCOM_STEPS:-4}" "${WR_LOCOM_SAMPLE_TOKENS:-1024}" "${WR_LOCOM_INNER_LR:-0.1}" \
    "${WR_LOCOM_PROX:-0.1}" "${WR_LOCOM_ALPHA:-1.0}" "${WR_LOCOM_NORM_CAP:-0.20}" \
    "${WR_LOCOM_NORM_TO_BASE:-0}" "${WR_LOCOM_MIN_COS_DESC:-0.0}" "${WR_LOCOM_REQUIRE_LOSS_DECREASE:-1}"
} | tee "${status_file}"

rc=0
env \
  WR_LOCOM_GENERATED_SCRIPT="${generated}" \
  WR_TRAIN_STEPS="${steps}" \
  WR_SCHEDULE_STEPS="${schedule_steps}" \
  WR_SEED="${seed}" \
  NPROC_PER_NODE="${nproc}" \
  WR_LOCOM_ENABLED="${WR_LOCOM_ENABLED:-1}" \
  WR_LOCOM_LAYERS="${WR_LOCOM_LAYERS:-all}" \
  WR_LOCOM_ACTIVE_WINDOWS="${WR_LOCOM_ACTIVE_WINDOWS:-}" \
  WR_LOCOM_START_STEP="${WR_LOCOM_START_STEP:-0}" \
  WR_LOCOM_END_STEP="${WR_LOCOM_END_STEP:-1000000000}" \
  WR_LOCOM_INTERVAL="${WR_LOCOM_INTERVAL:-1}" \
  WR_LOCOM_STEPS="${WR_LOCOM_STEPS:-4}" \
  WR_LOCOM_SAMPLE_TOKENS="${WR_LOCOM_SAMPLE_TOKENS:-1024}" \
  WR_LOCOM_INNER_LR="${WR_LOCOM_INNER_LR:-0.1}" \
  WR_LOCOM_TARGET_GAMMA="${WR_LOCOM_TARGET_GAMMA:-1.0}" \
  WR_LOCOM_PROX="${WR_LOCOM_PROX:-0.1}" \
  WR_LOCOM_ALPHA="${WR_LOCOM_ALPHA:-1.0}" \
  WR_LOCOM_NORM_CAP="${WR_LOCOM_NORM_CAP:-0.20}" \
  WR_LOCOM_NORM_TO_BASE="${WR_LOCOM_NORM_TO_BASE:-0}" \
  WR_LOCOM_MIN_COS_DESC="${WR_LOCOM_MIN_COS_DESC:-0.0}" \
  WR_LOCOM_REQUIRE_LOSS_DECREASE="${WR_LOCOM_REQUIRE_LOSS_DECREASE:-1}" \
  WR_LOCOM_LOG_STEPS="${WR_LOCOM_LOG_STEPS:-0,1,2,10,50,125,250,500,1000,1500,1750,1875,2000,2125,2250,2375,2500,2625,2750,2875,3000,3030,3040}" \
  SCREEN_VAL_EVERY="${SCREEN_VAL_EVERY:-125}" \
  bash tools/run_wr_record_locoprop_m.sh > "${run_log}" 2>&1 || rc=$?

printf 'finished %s rc=%s\n' "$(date -Is)" "${rc}" | tee -a "${status_file}"
touch "/root/prime_track3_logs/${label}.DONE"
exit "${rc}"
