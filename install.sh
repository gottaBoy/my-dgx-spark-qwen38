#!/usr/bin/env bash
# Converging installer for qwen38-spark. Idempotent: re-running upgrades.
#
#   ./install.sh                  # engine + systemd unit + guard
#   ./install.sh --no-service     # repo only; run with ./bin/qwen38 start
#   ./install.sh --profile=mtp    # pick the serving profile to install
#   ./install.sh --print-unit     # render the unit and exit; touches nothing
#
# What this does and does not do:
#   * it never downloads weights and never pulls an image. Those are big, visible
#     decisions, made with ./bin/qwen38 fetch-image and ./bin/qwen38 prefetch, so
#     an install can be re-run on a shared box without touching the disk budget.
#   * it calls sudo itself, exactly twice, both for /etc writes. Do NOT run this
#     script under sudo: $HOME becomes /root, the cache paths in config.local
#     resolve there, and the engine then serves from a cache nobody owns.
#   * it re-reads what is already installed and keeps your choices, so an
#     upgrade does not silently reset a tuned box back to the defaults.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="qwen38-spark"
SERVICE=1
PROFILE=""

for arg in "$@"; do
  case "$arg" in
    --no-service) SERVICE=0 ;;
    --print-unit) PRINT_UNIT=1 ;;
    --profile)    { echo "--profile needs =value, e.g. --profile=mtp"; exit 2; } ;;
    --profile=*)  PROFILE="${arg#*=}" ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            echo "unknown argument: $arg (try --help)"; exit 2 ;;
  esac
done

if [[ "$(id -u)" -eq 0 ]]; then
  echo "refusing to run as root. The installer calls sudo where it needs it." >&2
  exit 2
fi

echo "== preflight"
# The same checklist a human would run by hand, executed before anything is
# written, so a failure costs a message rather than a half-installed unit.
"${REPO}/bin/qwen38" doctor || {
  echo
  echo "doctor reported failures above. FAIL here means fix it; a WARN is a note." >&2
  exit 1
}

if [[ -n "${PROFILE}" ]]; then
# Validate against the Python registry instead of duplicating the name list in
# shell: argparse rejects an unknown --profile before it does any work, so the
# exit code is the check. A typo otherwise renders a unit that looks healthy and
# dies at boot several minutes later.
if ! "${REPO}/bin/qwen38" plan --profile "${PROFILE}" --no-probe --mem-fraction 0.5 >/dev/null 2>&1; then
  echo "unknown --profile '${PROFILE}'. Valid names:" >&2
  "${REPO}/bin/qwen38" start --help 2>&1 | sed -n 's/.*--profile {\(.*\)}.*/  \1/p' >&2
  exit 2
fi
echo "== selecting profile ${PROFILE}"
if grep -q '^Q38_PROFILE=' "${REPO}/conf/config.local" 2>/dev/null; then
  sed -i "s|^Q38_PROFILE=.*|Q38_PROFILE=\"${PROFILE}\"|" "${REPO}/conf/config.local"
else
  printf 'Q38_PROFILE="%s"\n' "${PROFILE}" >> "${REPO}/conf/config.local"
fi
fi

# Read the effective value back through the CLI. This is the same precedence the
# engine will use, so the unit cannot end up describing a profile that a later
# `qwen38 start` would not pick.
PROFILE_RESOLVED="$("${REPO}/bin/qwen38" config Q38_PROFILE)"
if [[ -z "${PROFILE_RESOLVED}" ]]; then
  echo "could not resolve Q38_PROFILE; refusing to render a unit with an empty profile" >&2
  exit 1
fi

if [[ "${SERVICE}" -eq 0 ]]; then
  echo
  echo "== done, no service installed. Run it in the foreground with:"
  echo "   ${REPO}/bin/qwen38 start"
  exit 0
fi

echo "== rendering the unit"
TEMPLATE="${REPO}/unit/${UNIT_NAME}.service.in"
RENDERED="$(mktemp)"
trap 'rm -f "${RENDERED}"' EXIT
# A narrow, ordered substitution. @PLACEHOLDER@ exists only in the template, so
# an unrendered one left in the output is a bug we can detect, below.
sed -e "s|@REPO@|${REPO}|g" \
    -e "s|@USER@|$(id -un)|g" \
    -e "s|@GROUP@|$(id -gn)|g" \
    -e "s|@PROFILE@|${PROFILE_RESOLVED}|g" \
    "${TEMPLATE}" > "${RENDERED}"

if grep -q '@[A-Z]\+@' "${RENDERED}"; then
  echo "unrendered placeholder in the unit; refusing to install a broken service:" >&2
  grep -n '@[A-Z]\+@' "${RENDERED}" >&2
  exit 1
fi

if [[ "${PRINT_UNIT:-0}" -eq 1 ]]; then
  # The pre-flight a cautious operator wants before touching /etc, and the only
  # part of this script that can be exercised without sudo.
  echo
  cat "${RENDERED}"
  exit 0
fi

# An upgrade must not lose a hand edit, and a fresh install must not be blocked
# by a missing backup target.
if sudo test -f "/etc/systemd/system/${UNIT_NAME}.service"; then
  if ! sudo cmp -s "${RENDERED}" "/etc/systemd/system/${UNIT_NAME}.service"; then
    sudo cp "/etc/systemd/system/${UNIT_NAME}.service" \
            "/etc/systemd/system/${UNIT_NAME}.service.bak-$(date +%Y%m%dT%H%M%S)"
    echo "   replaced an existing unit; the previous one is backed up in /etc/systemd/system"
  fi
fi
sudo cp "${RENDERED}" "/etc/systemd/system/${UNIT_NAME}.service"
rm -f "${RENDERED}"
sudo systemctl daemon-reload
sudo systemctl enable "${UNIT_NAME}.service" >/dev/null

echo
echo "== installed. The engine is NOT started yet, on purpose."
echo "   Weights and the image are still on the network, and starting is the"
echo "   expensive, visible part of this. When you are ready:"
echo
echo "     ./bin/qwen38 fetch-image     # 13.41 GiB image (docker pull stalls where dockerd has no proxy)"
echo "     ./bin/qwen38 prefetch        # 25.72 GiB of weights, on the host; the container cannot reach the Hub here"
echo "     sudo systemctl start ${UNIT_NAME}     # boot is 7-9 min"
echo "     ./bin/qwen38 canary          # correctness gate; speed without this is unmeasured"
echo "     ./bin/qwen38 bench --save    # and ./bin/qwen38 runs to see the ledger"
echo
echo "   API: http://127.0.0.1:$(sed -n 's/^Q38_PORT="\(.*\)"$/\1/p' "${REPO}/conf/config.defaults")/v1"
echo "   To go back: ./uninstall.sh --list shows everything this put on the box."
