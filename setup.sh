#!/usr/bin/env bash
# =============================================================================
# RLAIF Trading - Setup Script
# Detects hardware, installs dependencies, configures environment
# =============================================================================
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helpers
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# Track errors but don't exit on every failure
ERRORS=0
trap 'if [ $ERRORS -gt 0 ]; then warn "$ERRORS non-critical errors occurred (see above)"; fi' EXIT

# =============================================================================
header "RLAIF Trading Setup"
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
info "Project directory: $PROJECT_DIR"

# =============================================================================
header "Detecting System"
# =============================================================================

# OS Detection
OS="$(uname -s)"
case "$OS" in
    Darwin) info "Operating System: macOS" ;;
    Linux)  warn "Linux detected — MLX features will be unavailable (requires macOS + Apple Silicon)" ;;
    *)      error "Unsupported OS: $OS"; exit 1 ;;
esac

# Architecture / Apple Silicon Detection
ARCH="$(uname -m)"
IS_APPLE_SILICON=false
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    IS_APPLE_SILICON=true
    # Detect chip name
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    success "Apple Silicon detected: $CHIP"
elif [[ "$OS" == "Darwin" ]]; then
    warn "Intel Mac detected — MLX requires Apple Silicon (M1/M2/M3/M4/M5)"
fi

# RAM Detection
if [[ "$OS" == "Darwin" ]]; then
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$((RAM_BYTES / 1073741824))
else
    RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
    RAM_GB=$((RAM_KB / 1048576))
fi
info "System RAM: ${RAM_GB} GB"

# Python Detection
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 11 ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    error "Python 3.11+ is required but not found."
    error "Install via: brew install python@3.12"
    exit 1
fi
success "Python: $($PYTHON_CMD --version)"

# =============================================================================
header "Virtual Environment"
# =============================================================================

VENV_DIR="$PROJECT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Activated: $(which python)"

# Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"

# =============================================================================
header "Installing Core Dependencies"
# =============================================================================

info "Installing core packages (numpy, pandas, scipy, pyyaml, etc.)..."
pip install -q \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scipy>=1.10.0" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0.0" \
    "click>=8.1.0" \
    "tqdm>=4.66.0" \
    "requests>=2.31.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.1.0" \
    "ta>=0.11.0" \
    2>&1 | tail -1 || { warn "Some core deps had issues"; ERRORS=$((ERRORS+1)); }
success "Core dependencies installed"

# =============================================================================
header "Installing ML Dependencies"
# =============================================================================

info "Installing PyTorch..."
if [[ "$IS_APPLE_SILICON" == true ]]; then
    pip install -q "torch>=2.0.0" 2>&1 | tail -1 || { warn "torch install had issues"; ERRORS=$((ERRORS+1)); }
else
    pip install -q "torch>=2.0.0" 2>&1 | tail -1 || { warn "torch install had issues"; ERRORS=$((ERRORS+1)); }
fi
success "PyTorch installed"

info "Installing transformers & sentence-transformers..."
pip install -q \
    "transformers>=4.30.0" \
    "sentence-transformers>=2.2.0" \
    "stable-baselines3>=2.0.0" \
    2>&1 | tail -1 || { warn "ML deps had issues"; ERRORS=$((ERRORS+1)); }
success "ML dependencies installed"

# =============================================================================
header "Installing LLM Cloud Dependencies"
# =============================================================================

info "Installing Anthropic / LangChain..."
pip install -q \
    "anthropic>=0.18.0" \
    "langchain>=0.1.0" \
    "langchain-anthropic>=0.1.0" \
    "langchain-community>=0.0.20" \
    "tiktoken>=0.5.0" \
    2>&1 | tail -1 || { warn "LLM cloud deps had issues"; ERRORS=$((ERRORS+1)); }
success "LLM cloud dependencies installed"

# =============================================================================
if [[ "$IS_APPLE_SILICON" == true ]]; then
    header "Installing MLX (Local LLM)"
    info "Installing mlx and mlx-lm for Apple Silicon..."
    pip install -q \
        "mlx>=0.20.0" \
        "mlx-lm>=0.20.0" \
        2>&1 | tail -1 || { warn "MLX install had issues"; ERRORS=$((ERRORS+1)); }
    success "MLX dependencies installed"
else
    info "Skipping MLX (requires Apple Silicon)"
fi

# =============================================================================
header "Installing Options & Broker Dependencies"
# =============================================================================

info "Installing options packages (yfinance, py_vollib)..."
pip install -q \
    "yfinance>=0.2.0" \
    "py_vollib>=1.0.1" \
    2>&1 | tail -1 || { warn "Options deps had issues"; ERRORS=$((ERRORS+1)); }
success "Options dependencies installed"

info "Installing broker packages (alpaca-py)..."
pip install -q \
    "alpaca-py>=0.20.0" \
    2>&1 | tail -1 || { warn "Broker deps had issues"; ERRORS=$((ERRORS+1)); }
success "Broker dependencies installed"

# =============================================================================
header "Installing Data & Monitoring Dependencies"
# =============================================================================

info "Installing data packages..."
pip install -q \
    "pyarrow>=12.0.0" \
    "fastparquet>=2023.4.0" \
    2>&1 | tail -1 || { warn "Data deps had issues"; ERRORS=$((ERRORS+1)); }

info "Installing monitoring packages..."
pip install -q \
    "evidently>=0.4.0" \
    "prometheus-client>=0.19.0" \
    "python-json-logger>=2.0.7" \
    "mlflow>=2.10.0" \
    2>&1 | tail -1 || { warn "Monitoring deps had issues"; ERRORS=$((ERRORS+1)); }

info "Installing API packages..."
pip install -q \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    2>&1 | tail -1 || { warn "API deps had issues"; ERRORS=$((ERRORS+1)); }
success "Data, monitoring & API dependencies installed"

# =============================================================================
header "Installing Dev Dependencies"
# =============================================================================

info "Installing dev packages..."
pip install -q \
    "jupyter>=1.0.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "plotly>=5.18.0" \
    "pytest>=7.4.0" \
    "pytest-cov>=4.1.0" \
    "pytest-asyncio>=0.23.0" \
    "black>=24.0.0" \
    "ruff>=0.2.0" \
    "mypy>=1.8.0" \
    2>&1 | tail -1 || { warn "Dev deps had issues"; ERRORS=$((ERRORS+1)); }
success "Dev dependencies installed"

# =============================================================================
header "Skipped Dependencies (manual install if needed)"
# =============================================================================

warn "Skipped: ta-lib (requires C library: brew install ta-lib)"
warn "Skipped: feast[redis] (heavy, needs Redis)"
warn "Skipped: faiss-gpu (needs CUDA)"
warn "Skipped: selenium (needs ChromeDriver)"
warn "Skipped: wandb (optional experiment tracking)"
warn "Skipped: optuna (optional hyperparameter tuning)"
info "Install any of these manually if needed."

# =============================================================================
header "Environment Configuration"
# =============================================================================

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        success "Created .env from .env.example"
        info "Edit .env with your API keys and settings"
    else
        warn ".env.example not found — cannot create .env"
    fi
else
    info ".env already exists — skipping"
fi

# =============================================================================
header "Model Recommendation"
# =============================================================================

RECOMMENDED_MODEL=""
if [[ "$IS_APPLE_SILICON" == true ]]; then
    if   [[ $RAM_GB -ge 128 ]]; then
        RECOMMENDED_MODEL="mlx-community/Qwen2.5-72B-Instruct-4bit"
        info "RAM: ${RAM_GB}GB → Recommended model: ${BOLD}Qwen2.5-72B-Instruct-4bit${NC}"
    elif [[ $RAM_GB -ge 64 ]]; then
        RECOMMENDED_MODEL="mlx-community/Qwen2.5-32B-Instruct-4bit"
        info "RAM: ${RAM_GB}GB → Recommended model: ${BOLD}Qwen2.5-32B-Instruct-4bit${NC}"
    elif [[ $RAM_GB -ge 32 ]]; then
        RECOMMENDED_MODEL="mlx-community/Qwen2.5-14B-Instruct-4bit"
        info "RAM: ${RAM_GB}GB → Recommended model: ${BOLD}Qwen2.5-14B-Instruct-4bit${NC}"
    else
        RECOMMENDED_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
        info "RAM: ${RAM_GB}GB → Recommended model: ${BOLD}Qwen2.5-7B-Instruct-4bit${NC}"
    fi

    # Update .env with recommended model if it exists
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        if grep -q "^MLX_MODEL=" "$PROJECT_DIR/.env" 2>/dev/null; then
            sed -i '' "s|^MLX_MODEL=.*|MLX_MODEL=$RECOMMENDED_MODEL|" "$PROJECT_DIR/.env" 2>/dev/null || true
        fi
    fi

    echo ""
    read -r -p "$(echo -e "${CYAN}Download recommended model now? (~4-16GB) [y/N]:${NC} ")" DOWNLOAD_MODEL
    if [[ "$DOWNLOAD_MODEL" =~ ^[Yy]$ ]]; then
        info "Downloading $RECOMMENDED_MODEL ..."
        info "This may take a while depending on your connection..."
        python -c "
from mlx_lm import load
print('Downloading and caching model...')
model, tokenizer = load('$RECOMMENDED_MODEL')
print('Model downloaded and verified!')
" 2>&1 || { warn "Model download failed — you can download later with: python -m mlx_lm.manage --model $RECOMMENDED_MODEL"; ERRORS=$((ERRORS+1)); }
        success "Model downloaded"
    else
        info "Skipping download. Download later with:"
        echo "  python -c \"from mlx_lm import load; load('$RECOMMENDED_MODEL')\""
    fi
else
    info "MLX models require Apple Silicon — using Claude API (cloud) instead"
    info "Set ANTHROPIC_API_KEY in .env to use Claude"
fi

# =============================================================================
header "Validation"
# =============================================================================

info "Checking key imports..."
VALIDATION_PASSED=true

check_import() {
    local pkg="$1"
    local display="${2:-$1}"
    if python -c "import $pkg" 2>/dev/null; then
        success "$display"
    else
        warn "$display — not available"
        VALIDATION_PASSED=false
    fi
}

check_import "numpy"              "numpy"
check_import "pandas"             "pandas"
check_import "scipy"              "scipy"
check_import "torch"              "torch"
check_import "transformers"       "transformers"
check_import "anthropic"          "anthropic"
check_import "langchain"          "langchain"
check_import "yfinance"           "yfinance"
check_import "fastapi"            "fastapi"
check_import "pydantic"           "pydantic"
check_import "click"              "click"

if [[ "$IS_APPLE_SILICON" == true ]]; then
    check_import "mlx"            "mlx"
    check_import "mlx_lm"        "mlx-lm"
fi

echo ""
if [[ "$VALIDATION_PASSED" == true ]]; then
    success "All key packages validated!"
else
    warn "Some packages could not be imported (see above)"
fi

# =============================================================================
header "Setup Complete!"
# =============================================================================

echo -e "${GREEN}${BOLD}RLAIF Trading is ready to go!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Edit your config:          vim .env"
echo "  3. Run the CLI:               python -m src.cli --help"
if [[ "$IS_APPLE_SILICON" == true ]]; then
    echo "  4. Test local LLM:            python -c \"from mlx_lm import load, generate; m,t = load('$RECOMMENDED_MODEL'); print(generate(m,t,'Hello'))\""
fi
echo ""
echo -e "${CYAN}System: $ARCH | RAM: ${RAM_GB}GB | Python: $(python --version 2>&1)${NC}"
if [[ "$IS_APPLE_SILICON" == true && -n "$RECOMMENDED_MODEL" ]]; then
    echo -e "${CYAN}Recommended MLX Model: $RECOMMENDED_MODEL${NC}"
fi
echo ""
