#!/usr/bin/env bash
set -euo pipefail

HOST="linux.ferdowsi.cloud"
USER="ubuntu"
PORT="2388"
LOCAL_DIR="/mnt/d/VeriLogos/historical_data/dataset2"
REMOTE_DIR='~/datasets/pheme_raw'
SSH_OPTS=(
  -i /home/lemi/.ssh/id_ed25519_verilogos
  -F /dev/null
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
)

FILES=(
  "6392078.zip"
  "phemernrdataset.tar.bz2"
  "phemerumourschemedataset.tar.bz2"
)

run_ssh() {
  ssh "${SSH_OPTS[@]}" -p "$PORT" "$USER@$HOST" "$@"
}

run_rsync() {
  local src="$1"
  local dst="$2"
  rsync -avP --inplace --append-verify -e "ssh -i /home/lemi/.ssh/id_ed25519_verilogos -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p $PORT" "$src" "$dst"
}

upload_one() {
  local name="$1"
  local local_path="$LOCAL_DIR/$name"
  local local_size
  local_size=$(stat -c '%s' "$local_path")

  run_ssh "mkdir -p $REMOTE_DIR"

  local remote_file="~/datasets/pheme_raw/$name"
  local remote_size
  remote_size=$(run_ssh "if [ -f $remote_file ]; then stat -c '%s' $remote_file; else echo 0; fi")
  if [ "$remote_size" != "0" ] && [ "$remote_size" != "$local_size" ]; then
    echo "[INFO] Removing partial remote file: $remote_file (remote=$remote_size local=$local_size)"
    run_ssh "rm -f $remote_file"
  fi

  local attempt=1
  local max_attempts=2
  while [ "$attempt" -le "$max_attempts" ]; do
    echo "[INFO] Uploading $name (attempt $attempt/$max_attempts)"
    if run_rsync "$local_path" "$USER@$HOST:$REMOTE_DIR/"; then
      remote_size=$(run_ssh "stat -c '%s' $remote_file")
      if [ "$remote_size" = "$local_size" ]; then
        echo "[OK] Size verified for $name: $remote_size bytes"
        return 0
      fi
      echo "[WARN] Size mismatch for $name: remote=$remote_size local=$local_size"
    else
      echo "[WARN] rsync failed for $name on attempt $attempt"
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -le "$max_attempts" ]; then
      echo "[INFO] Retrying $name after cleanup"
      run_ssh "rm -f $remote_file"
      sleep 2
    fi
  done

  echo "[ERROR] Upload failed after retries: $name" >&2
  return 1
}

main() {
  echo "[STEP] SSH preflight"
  if ! run_ssh "echo connected"; then
    echo "[ERROR] SSH preflight failed for $USER@$HOST:$PORT (authentication or connectivity issue)" >&2
    exit 1
  fi

  echo "[STEP] Validating local dataset files"
  for name in "${FILES[@]}"; do
    if [ ! -f "$LOCAL_DIR/$name" ]; then
      echo "[ERROR] Missing local file: $LOCAL_DIR/$name" >&2
      exit 1
    fi
  done

  echo "[STEP] Self-test upload (dummy file)"
  local dummy
  dummy=$(mktemp /tmp/pheme_upload_dummy_XXXX.txt)
  echo "pheme-upload-self-test $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dummy"
  run_ssh "mkdir -p $REMOTE_DIR"
  run_rsync "$dummy" "$USER@$HOST:$REMOTE_DIR/"
  local dummy_base
  dummy_base=$(basename "$dummy")
  run_ssh "test -f ~/datasets/pheme_raw/$dummy_base"
  run_ssh "rm -f ~/datasets/pheme_raw/$dummy_base"
  rm -f "$dummy"
  echo "[OK] Dummy upload test passed"

  echo "[STEP] Uploading archives"
  for name in "${FILES[@]}"; do
    upload_one "$name"
  done

  echo "[STEP] Remote listing"
  run_ssh "ls -lh ~/datasets/pheme_raw"

  echo "[DONE] Upload pipeline complete"
}

main "$@"
