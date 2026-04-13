#!/usr/bin/env bash
set -euo pipefail

echo "=== neuroLoop setup ==="

# ------------------------------------------------------------------
# 0. Load .env if it exists
# ------------------------------------------------------------------
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "Loaded .env"
else
  echo "No .env found — copy .env.example to .env and fill in your AWS credentials"
fi

# ------------------------------------------------------------------
# 1. Install uv if not present
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

# ------------------------------------------------------------------
# 2. Create Python venv and install all Python deps
# ------------------------------------------------------------------
echo "Creating Python environment..."
uv venv .venv --python 3.11
source .venv/bin/activate

echo "Installing tribev2 + backend deps..."
uv pip install -e ".[plotting]"
uv pip install -r dashboard/backend/requirements.txt

# ------------------------------------------------------------------
# 3. Install Node.js via nvm if not present, then frontend deps
# ------------------------------------------------------------------
if ! command -v node &>/dev/null; then
  echo "Installing Node.js..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm install --lts
fi
echo "node $(node --version)"

echo "Installing frontend dependencies..."
cd dashboard/frontend
npm install
cd ../..

# ------------------------------------------------------------------
# 4. Pre-download the TRIBE v2 model from HuggingFace
# ------------------------------------------------------------------
echo "Downloading TRIBE v2 model (this may take a few minutes on first run)..."
uv pip install huggingface-hub  # ensure CLI is available
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/tribev2', 'config.yaml', local_dir='./cache/facebook/tribev2')
hf_hub_download('facebook/tribev2', 'best.ckpt', local_dir='./cache/facebook/tribev2')
print('Model downloaded to ./cache/')
"

# ------------------------------------------------------------------
# 5. Pre-fetch fsaverage5 mesh (small, avoids delay on first request)
# ------------------------------------------------------------------
echo "Downloading fsaverage5 mesh..."
python -c "from nilearn.datasets import fetch_surf_fsaverage; fetch_surf_fsaverage('fsaverage5')"

# ------------------------------------------------------------------
# 6. Start both servers
# ------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "Starting servers..."
echo "  Backend:  http://0.0.0.0:8000"
echo "  Frontend: http://localhost:5173"
echo ""

# Start backend in background
cd dashboard/backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ../..

# Start frontend (foreground)
cd dashboard/frontend
npm run dev -- --host 0.0.0.0

# Clean up backend on exit
kill $BACKEND_PID 2>/dev/null
