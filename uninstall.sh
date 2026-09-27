#!/usr/bin/env bash
# Remove what install.sh put on the box. Read-only by default.
#
#   ./uninstall.sh --list     # show what would be touched, change nothing
#   ./uninstall.sh            # remove the unit and the container
#   ./uninstall.sh --purge    # also delete the weight and compile caches (~24 GB)
#   ./uninstall.sh --user     # remove the current user's unit, without sudo
#
# The ledger under state/evidence is never deleted by either mode. It is the
# measurement record; if you are removing the stack you may still want the
# numbers, and a run history you cannot recover is a bad trade for one directory.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="qwen38-spark"
CONTAINER="qwen38-spark"
LIST_ONLY=0
PURGE=0
USER_SERVICE=0

for arg in "$@"; do
  case "$arg" in
    --list)   LIST_ONLY=1 ;;
    --purge)  PURGE=1 ;;
    --user)   USER_SERVICE=1 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)        echo "unknown argument: $arg (try --help)"; exit 2 ;;
  esac
done

if [[ "$(id -u)" -eq 0 ]]; then
  echo "refusing to run as root; this script calls sudo for the /etc writes only." >&2
  exit 2
fi

CACHE_DIR="$("${REPO}/bin/qwen38" config Q38_CACHE_DIR)"
CACHE_DIR="${CACHE_DIR/#\~/${HOME}}"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}.service"
if [[ "${USER_SERVICE}" -eq 1 ]]; then
  UNIT_PATH="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user/${UNIT_NAME}.service"
fi

echo "== what this repo put on the box"
EXISTS=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
if [[ -f "${UNIT_PATH}" ]]; then
  note present "${UNIT_PATH}"; EXISTS=1
else
  note absent  "${UNIT_PATH}"
fi
for backup in "${UNIT_PATH}".bak-*; do
  [[ -e "${backup}" ]] || continue
  note present "${backup} (unit from an earlier install)"; EXISTS=1
done
if [[ -n "$(docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}' 2>/dev/null)" ]]; then
  note present "container ${CONTAINER}"; EXISTS=1
else
  note absent  "container ${CONTAINER}"
fi
if [[ -d "${CACHE_DIR}" ]]; then
  note present "weights and caches ${CACHE_DIR} ($(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1))"
else
  note absent  "weights and caches ${CACHE_DIR}"
fi
if [[ -d "${REPO}/state/evidence" ]] && [[ -n "$(ls -A "${REPO}/state/evidence" 2>/dev/null)" ]]; then
  note kept    "run ledger ${REPO}/state/evidence (never removed)"
fi

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  echo
  echo "== --list: nothing was changed."
  exit 0
fi

echo
if [[ "${EXISTS}" -eq 0 && "${PURGE}" -eq 0 ]]; then
  echo "nothing to remove (no unit, no container). Pass --purge for the caches."
  exit 0
fi

if [[ -f "${UNIT_PATH}" ]]; then
  echo "== stopping and removing the unit"
  if [[ "${USER_SERVICE}" -eq 1 ]]; then
    systemctl --user stop "${UNIT_NAME}.service" 2>/dev/null || true
    systemctl --user disable "${UNIT_NAME}.service" 2>/dev/null || true
    rm -f "${UNIT_PATH}"
    systemctl --user daemon-reload
  else
    sudo systemctl stop "${UNIT_NAME}.service" 2>/dev/null || true
    sudo systemctl disable "${UNIT_NAME}.service" 2>/dev/null || true
    sudo rm -f "${UNIT_PATH}"
    sudo systemctl daemon-reload
  fi
fi

if [[ -n "$(docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}' 2>/dev/null)" ]]; then
  echo "== removing the container"
  docker rm -f "${CONTAINER}" >/dev/null
fi

if [[ "${PURGE}" -eq 1 && -d "${CACHE_DIR}" ]]; then
  echo "== purging ${CACHE_DIR}"
  rm -rf "${CACHE_DIR}"
else
  echo "   caches left in place at ${CACHE_DIR}; add --purge to delete them."
fi

echo
echo "done. The repo itself, conf/config.local and state/evidence are untouched."
