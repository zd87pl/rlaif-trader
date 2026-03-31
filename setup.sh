#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--minimal|--full] [--apple-local] [--yes]

Modes:
  --minimal      Install the smallest useful local dev environment
                 (editable package + dev + options + broker + data + api)
  --full         Install the broader stack including llm-cloud and monitoring
  --apple-local  Also install llm-local extras (MLX) on Apple Silicon only
  --yes          Non-interactive; do not prompt

Examples:
  ./setup.sh --minimal
  ./setup.sh --full --yes
  ./setup.sh --full --apple-local
EOF
}

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODE="minimal"
INSTALL_APPLE_LOCAL=false
NON_INTERACTIVE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minimal) MODE="minimal" ;;
    --full) MODE="full" ;;
    --apple-local) INSTALL_APPLE_LOCAL=true ;;
    --yes) NON_INTERACTIVE=true ;;
    -h|--help) usage; exit 0 ;;
    *) error "Unknown argument: $1"; usage; exit 1 ;;
  esac
  shift
done

header "RLAIF Trading Local Setup"
info "Project directory: $PROJECT_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"
IS_APPLE_SILICON=false
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  IS_APPLE_SILICON=true
fi
info "Detected platform: $OS / $ARCH"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
  if command -v "$cmd" >/dev/null 2>&1; then
    PY_VER=$("$cmd" --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -gt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 11 ) ]]; then
      PYTHON_CMD="$cmd"
      break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  error "Python 3.11+ is required but was not found on PATH."
  if [[ "$OS" == "Darwin" ]]; then
    echo "Install with: brew install python@3.11"
  fi
  exit 1
fi
success "Using Python: $($PYTHON_CMD --version)"

VENV_DIR="$PROJECT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  header "Creating Virtual Environment"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
  success "Created $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Activated: $(which python)"

header "Upgrading Packaging Tooling"
python -m pip install --upgrade pip setuptools wheel

header "Installing Project"
BASE_EXTRAS="dev,options,broker,data,api"
if [[ "$MODE" == "full" ]]; then
  BASE_EXTRAS="${BASE_EXTRAS},llm-cloud,monitoring"
fi

info "Installing editable package with extras: [$BASE_EXTRAS]"
pip install -e ".[${BASE_EXTRAS}]"

if [[ "$INSTALL_APPLE_LOCAL" == true ]]; then
  if [[ "$IS_APPLE_SILICON" == true ]]; then
    info "Installing Apple Silicon local LLM extras: [llm-local]"
    pip install -e ".[llm-local]"
  else
    warn "Skipping llm-local extras: not Apple Silicon"
  fi
fi

header "Preparing Environment File"
if [[ ! -f "$PROJECT_DIR/.env.example" ]]; then
  warn ".env.example missing; creating a sane default template"
  cat > "$PROJECT_DIR/.env.example" <<'EOF'
# Core API credentials
ANTHROPIC_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Runtime selection
LLM_BACKEND=claude
DEVICE=cpu

# Optional local MLX model (Apple Silicon only)
MLX_MODEL=
EOF
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  success "Created .env from .env.example"
else
  info ".env already exists; leaving it untouched"
fi

header "Validation"
python - <<'PY'
import importlib
import sys

checks = [
    ("yaml", "PyYAML"),
    ("click", "click"),
    ("pandas", "pandas"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("requests", "requests"),
    ("fastapi", "fastapi"),
    ("pytest", "pytest"),
]

failures = []
for module, label in checks:
    try:
        importlib.import_module(module)
        print(f"[OK] {label}")
    except Exception as exc:
        print(f"[WARN] {label}: {exc}")
        failures.append(label)

try:
    import main  # noqa: F401
    print("[OK] main import")
except Exception as exc:
    print(f"[WARN] main import: {exc}")
    failures.append("main import")

if failures:
    print("\nValidation completed with warnings:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(0)

print("\nValidation completed successfully.")
PY

header "Next Steps"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Fill in any API keys in .env"
echo "3. Check system status: python -m src.cli status"
echo "4. Run smoke tests: python -m pytest tests/test_pipeline.py -q -o addopts=''"

if [[ "$NON_INTERACTIVE" == false ]]; then
  echo ""
  info "Setup complete."
fi
