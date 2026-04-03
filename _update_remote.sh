#!/bin/bash
set -e

# Stop any running app
pkill -f "python app.py" 2>/dev/null || true
sleep 1

# Replace old code with updated version
rm -rf /home/blackmotion
cp -r /home/blackmotion_new/blackmotion /home/blackmotion
rm -rf /home/blackmotion_new

# Patch for public access
cd /home/blackmotion
sed -i 's/demo\.launch()/demo.launch(server_name="0.0.0.0", server_port=7860, share=True)/' app.py

echo "=== Verify ==="
head -3 app.py
grep "demo.launch" app.py

# Install OSMesa/EGL for PyVista offscreen rendering (headless GPU)
apt-get install -y -qq libgl1-mesa-glx libegl1-mesa libegl-dev libosmesa6 libosmesa6-dev > /dev/null 2>&1 || true

# Launch
source /home/venv/bin/activate
export HF_TOKEN="$1"
export PYTHONIOENCODING=utf-8

script -q -c "/home/venv/bin/python app.py" /home/blackmotion.log &
echo "App launching... waiting 40s"
sleep 40

# Show public URL
grep -oP 'https://[a-z0-9]+\.gradio\.live' /home/blackmotion.log || echo "No share URL found yet"
echo "=== Last 10 log lines ==="
tail -10 /home/blackmotion.log
