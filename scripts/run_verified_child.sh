#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "usage: $0 REPORT_JSON TOP_LEVEL_KEY EXPECTED_VALUE MAX_SECONDS -- COMMAND [ARGS...]" >&2
  exit 2
fi
report=$1
key=$2
expected=$3
max_seconds=$4
shift 4
if [[ "${1:-}" != "--" ]]; then
  echo "missing -- before child command" >&2
  exit 2
fi
shift
if [[ "$#" -eq 0 ]]; then
  echo "missing child command" >&2
  exit 2
fi

rm -f "$report"
mkdir -p "$(dirname "$report")"
setsid "$@" &
child_pid=$!
verified=0
loops=$((max_seconds * 4))
for _ in $(seq 1 "$loops"); do
  if python - "$report" "$key" "$expected" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path
path, key, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not path.exists():
    raise SystemExit(1)
value = json.loads(path.read_text()).get(key)
raise SystemExit(0 if str(value) == expected else 1)
PY
  then
    verified=1
    break
  fi
  if ! kill -0 "$child_pid" 2>/dev/null; then
    break
  fi
  sleep 0.25
done

if [[ "$verified" -ne 1 ]]; then
  echo "verified child failed to produce $report with $key=$expected" >&2
  kill -KILL -- -"$child_pid" 2>/dev/null || true
  disown "$child_pid" 2>/dev/null || true
  exit 1
fi

if kill -0 "$child_pid" 2>/dev/null; then
  echo "verified $report; terminating stale native-runtime teardown group" >&2
  kill -KILL -- -"$child_pid" 2>/dev/null || true
fi
disown "$child_pid" 2>/dev/null || true
echo "verified $report ($key=$expected)"
