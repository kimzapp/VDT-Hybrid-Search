#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Script: zip từng thư mục -> upload Google Drive -> lấy public link
#        -> xoá file zip local -> lưu link vào text file
# ============================================================

# Đường dẫn rclone binary của bạn
RCLONE_BIN="$HOME/rclone-v1.74.3-linux-amd64/rclone"

# File chứa danh sách thư mục cần zip, mỗi dòng là một thư mục
FOLDER_LIST_FILE="${1:-folders.txt}"

# Tên remote rclone đã config
REMOTE_NAME="${2:-gdrive}"

# Thư mục đích trên Google Drive
REMOTE_DIR="${3:-upload_artifacts}"

# File output lưu public links
OUTPUT_LINK_FILE="${4:-public_links.txt}"

# Thư mục tạm để chứa file zip trước khi upload
TMP_DIR="${TMPDIR:-/tmp}/rclone_zip_upload"

# Mức nén zip: 0-9
ZIP_LEVEL=6

mkdir -p "$TMP_DIR"

echo "============================================================"
echo "Folder list file : $FOLDER_LIST_FILE"
echo "Rclone binary    : $RCLONE_BIN"
echo "Rclone remote    : ${REMOTE_NAME}:"
echo "Remote directory : $REMOTE_DIR"
echo "Output link file : $OUTPUT_LINK_FILE"
echo "Temp zip dir     : $TMP_DIR"
echo "============================================================"
echo

# ------------------------------------------------------------
# Kiểm tra dependency
# ------------------------------------------------------------

if [[ ! -x "$RCLONE_BIN" ]]; then
    echo "ERROR: Không tìm thấy hoặc không có quyền chạy rclone tại:"
    echo "  $RCLONE_BIN"
    echo
    echo "Kiểm tra lại bằng:"
    echo "  ls -lh $RCLONE_BIN"
    echo
    echo "Nếu chưa có quyền execute, chạy:"
    echo "  chmod +x $RCLONE_BIN"
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "ERROR: Chưa cài zip."
    echo "Cài bằng:"
    echo "  sudo apt update && sudo apt install -y zip"
    exit 1
fi

if [[ ! -f "$FOLDER_LIST_FILE" ]]; then
    echo "ERROR: Không tìm thấy file danh sách thư mục:"
    echo "  $FOLDER_LIST_FILE"
    exit 1
fi

echo "Checking rclone version..."
"$RCLONE_BIN" version
echo

echo "Checking rclone remote..."
if ! "$RCLONE_BIN" lsd "${REMOTE_NAME}:" >/dev/null 2>&1; then
    echo "ERROR: Remote rclone không hoạt động hoặc không tồn tại:"
    echo "  ${REMOTE_NAME}:"
    echo
    echo "Kiểm tra bằng:"
    echo "  $RCLONE_BIN listremotes"
    echo "  $RCLONE_BIN lsd ${REMOTE_NAME}:"
    exit 1
fi

echo "Creating remote directory if needed..."
"$RCLONE_BIN" mkdir "${REMOTE_NAME}:${REMOTE_DIR}" >/dev/null 2>&1 || true
echo

# Ghi mới file link mỗi lần chạy
: > "$OUTPUT_LINK_FILE"

TOTAL=0
SUCCESS=0
SKIPPED=0
FAILED=0

# ------------------------------------------------------------
# Xử lý từng folder
# ------------------------------------------------------------

while IFS= read -r folder || [[ -n "$folder" ]]; do
    # Bỏ qua dòng trống
    if [[ -z "${folder// }" ]]; then
        continue
    fi

    # Bỏ qua comment bắt đầu bằng #
    if [[ "$folder" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

    # Xoá dấu / cuối path nếu có
    folder="${folder%/}"

    if [[ ! -d "$folder" ]]; then
        echo "WARNING: Bỏ qua vì không phải thư mục hợp lệ:"
        echo "  $folder"
        echo
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    parent_dir="$(dirname "$folder")"
    folder_name="$(basename "$folder")"

    zip_name="${folder_name}.zip"
    zip_path="${TMP_DIR}/${zip_name}"
    remote_path="${REMOTE_NAME}:${REMOTE_DIR}/${zip_name}"

    echo "============================================================"
    echo "Processing folder : $folder"
    echo "Zip name          : $zip_name"
    echo "Zip path          : $zip_path"
    echo "Remote path       : $remote_path"
    echo "============================================================"

    # Xoá zip cũ nếu còn tồn tại
    rm -f "$zip_path"

    echo "[1/4] Zipping folder..."
    if ! (
        cd "$parent_dir"
        zip -q -r "-${ZIP_LEVEL}" "$zip_path" "$folder_name"
    ); then
        echo "ERROR: Zip thất bại với thư mục:"
        echo "  $folder"
        echo
        FAILED=$((FAILED + 1))
        continue
    fi

    if [[ ! -f "$zip_path" ]]; then
        echo "ERROR: Không tìm thấy file zip sau khi nén:"
        echo "  $zip_path"
        echo
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "[2/4] Uploading to Google Drive..."
    if ! "$RCLONE_BIN" copyto "$zip_path" "$remote_path" -P; then
        echo "ERROR: Upload thất bại:"
        echo "  $zip_path"
        echo "File zip local được giữ lại để bạn có thể xử lý lại:"
        echo "  $zip_path"
        echo
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "[3/4] Creating public link..."
    public_link="$("$RCLONE_BIN" link "$remote_path" || true)"

    if [[ -z "$public_link" ]]; then
        echo "ERROR: Không lấy được public link cho:"
        echo "  $remote_path"
        echo "File zip local được giữ lại:"
        echo "  $zip_path"
        echo
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "${zip_name}: ${public_link}" >> "$OUTPUT_LINK_FILE"

    echo "[4/4] Removing local zip..."
    rm -f "$zip_path"

    SUCCESS=$((SUCCESS + 1))

    echo "DONE: $zip_name"
    echo "LINK: $public_link"
    echo

done < "$FOLDER_LIST_FILE"

echo "============================================================"
echo "Finished"
echo "Total   : $TOTAL"
echo "Success : $SUCCESS"
echo "Skipped : $SKIPPED"
echo "Failed  : $FAILED"
echo
echo "Public links saved to:"
echo "  $OUTPUT_LINK_FILE"
echo "============================================================"