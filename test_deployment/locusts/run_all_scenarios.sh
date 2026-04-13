#!/bin/bash
# ==============================================================================
# Запуск всіх 6 сценаріїв через Locust Web UI.
#
# Використання:
#   ./run_all_scenarios.sh [CPU_SERVICE_HOST]
#
# Приклад:
#   ./run_all_scenarios.sh http://localhost:8080
#
# Після запуску відкрий http://localhost:8089 для моніторингу.
# ==============================================================================

HOST="${1:-http://localhost:8080}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Dataset Generation: 6 scenarios"
echo "  Target: $HOST"
echo "  Locust UI: http://localhost:8089"
echo "============================================"

locust -f "${SCRIPT_DIR}/locustfile_scenarios.py" --host="$HOST"
