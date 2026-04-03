#!/bin/bash
# ============================================================
# Deploy Blackmotion (Video → Brain) on Jarvis Labs
# GPU: L4 (24GB VRAM) | Storage: 150GB | Region: IN2
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load env vars
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Validate
if [ -z "$JL_API_KEY" ]; then
    echo "ERROR: JL_API_KEY not set in .env"
    exit 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set in .env (needed for Meta LLaMA 3.2 weights)"
    echo "Get one at: https://huggingface.co/settings/tokens"
    exit 1
fi

export PYTHONIOENCODING=utf-8
export MSYS_NO_PATHCONV=1

echo "=========================================="
echo "  Deploying Blackmotion on Jarvis Labs"
echo "  GPU: L4 | Storage: 150GB | Region: IN2"
echo "=========================================="

# Step 1: Create instance
echo ""
echo "[1/4] Creating L4 instance (150GB storage)..."
INSTANCE_JSON=$(jl create --gpu L4 --storage 150 --json --yes)
MACHINE_ID=$(echo "$INSTANCE_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['machine_id'])")
echo "Instance created: $MACHINE_ID"

# Step 2: Upload project files
echo ""
echo "[2/4] Uploading project files..."
jl upload "$MACHINE_ID" "$SCRIPT_DIR/blackmotion" /home/blackmotion
jl upload "$MACHINE_ID" "$SCRIPT_DIR/tribev2" /home/tribev2
jl upload "$MACHINE_ID" "$SCRIPT_DIR/pyproject.toml" /home/pyproject.toml
jl upload "$MACHINE_ID" "$SCRIPT_DIR/README.md" /home/README.md

# Step 3: Install deps
echo ""
echo "[3/4] Installing dependencies..."
jl exec "$MACHINE_ID" -- bash -c "
set -e

# Fix nested directory structure from upload
mkdir -p /home/tribev2_pkg && cp -r /home/tribev2/tribev2/* /home/tribev2_pkg/
rm -rf /home/tribev2 && mv /home/tribev2_pkg /home/tribev2
mkdir -p /home/blackmotion_app && cp -r /home/blackmotion/blackmotion/* /home/blackmotion_app/
rm -rf /home/blackmotion && mv /home/blackmotion_app /home/blackmotion

# Install Python 3.12 (instance ships with 3.10, we need >=3.11)
apt-get update -qq
apt-get install -y -qq software-properties-common ffmpeg libsndfile1 git > /dev/null 2>&1
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt-get update -qq && apt-get install -y -qq python3.12 python3.12-venv python3.12-dev > /dev/null 2>&1

# Create venv
python3.12 -m venv /home/venv
source /home/venv/bin/activate

# Install PyTorch (CUDA 12.4 compatible with Jarvis Labs L4 driver)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

# Install tribev2 core deps
pip install numpy==2.2.6 x_transformers==1.27.20 moviepy transformers huggingface_hub einops pyyaml spacy soundfile julius gtts langdetect Levenshtein --quiet

# Install plotting deps
pip install nibabel matplotlib seaborn colorcet nilearn scipy scikit-image --quiet

# Install Gradio and extras
pip install 'gradio>=4.0.0' pandas Pillow --quiet

# Install tribev2 package
cd /home && pip install -e . --no-deps --quiet

# Verify
python -c 'from tribev2 import TribeModel; print(\"tribev2 OK\")'
python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}\")'

echo 'All dependencies installed successfully!'
"

# Step 4: Patch and launch
echo ""
echo "[4/4] Launching Blackmotion with public access..."
jl exec "$MACHINE_ID" -- bash -c "
source /home/venv/bin/activate
export HF_TOKEN=$HF_TOKEN
export PYTHONIOENCODING=utf-8

cd /home/blackmotion

# Patch Gradio for public access
sed -i 's/demo\.launch()/demo.launch(server_name=\"0.0.0.0\", server_port=7860, share=True)/' app.py

# Launch with full output capture
script -q -c 'python app.py' /home/blackmotion.log &
sleep 40

# Extract and display the public URL
PUBLIC_URL=\$(grep -oP 'https://[a-z0-9]+\.gradio\.live' /home/blackmotion.log)
echo ''
echo '=========================================='
echo '  Blackmotion is LIVE!'
echo \"  Public URL: \$PUBLIC_URL\"
echo \"  Instance ID: $MACHINE_ID\"
echo ''
echo '  Commands:'
echo \"    jl ssh $MACHINE_ID          # SSH in\"
echo \"    jl pause $MACHINE_ID --yes  # Stop billing\"
echo \"    jl destroy $MACHINE_ID --yes # Delete\"
echo '=========================================='
"
