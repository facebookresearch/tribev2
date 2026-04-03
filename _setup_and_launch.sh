#!/bin/bash
set -e

HF_TOKEN="$1"

# ── Step 1: System deps ──
echo "Setup Wardogz Brain Scanner v2..."
if ! command -v python3.12 &>/dev/null || [ ! -d /home/venv ]; then
    apt-get update -qq
    apt-get install -y -qq software-properties-common ffmpeg libsndfile1 git \
        libgl1-mesa-glx libegl1-mesa libegl-dev libosmesa6 libosmesa6-dev > /dev/null 2>&1
    add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev > /dev/null 2>&1

    # ── Step 2: Venv + PyTorch ──
    python3.12 -m venv /home/venv
    /home/venv/bin/pip install --upgrade pip --quiet
    /home/venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

    # ── Step 3: All deps ──
    /home/venv/bin/pip install numpy==2.2.6 x_transformers==1.27.20 moviepy transformers huggingface_hub einops pyyaml spacy soundfile julius gtts langdetect Levenshtein --quiet
    /home/venv/bin/pip install nibabel matplotlib seaborn colorcet nilearn scipy scikit-image pyvista vtk --quiet
    /home/venv/bin/pip install exca neuralset neuraltrain --quiet
    /home/venv/bin/pip install "gradio>=4.0.0" pandas Pillow --quiet

    # ── Step 4: tribev2 package ──
    cd /home && /home/venv/bin/pip install -e . --no-deps --quiet
fi

# ── Configure HF ──
/home/venv/bin/python -c "
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=False)
print('Meta Auth OK')
"

# ── Verify ──
/home/venv/bin/python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ── Update blackmotion code ──
pkill -f "python.*app.py" 2>/dev/null || true
kill -9 $(lsof -ti:7860) 2>/dev/null || true
if [ -d /home/blackmotion_new ]; then
    rm -rf /home/blackmotion
    cp -r /home/blackmotion_new/blackmotion /home/blackmotion
    rm -rf /home/blackmotion_new
fi

cd /home/blackmotion
sed -i 's/demo\.launch()/demo.launch(server_name="0.0.0.0", server_port=7860, share=True)/' app.py

# ── Launch ──
export HF_TOKEN="$HF_TOKEN"
export PYTHONIOENCODING=utf-8
nohup /home/venv/bin/python -u app.py > /home/blackmotion.log 2>&1 &
echo "App launching... waiting 40s"
sleep 40

grep -oP 'https://[a-z0-9]+\.gradio\.live' /home/blackmotion.log || echo "No share URL found yet"
echo "=== Last 10 log lines ==="
tail -10 /home/blackmotion.log
