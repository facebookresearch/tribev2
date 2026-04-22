<div align="center">

# neuroLoop

**Brain activity visualization dashboard powered by TRIBE v2**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

</div>

Upload a video, audio clip, or text — neuroLoop runs [TRIBE v2](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) to predict fMRI brain responses, then visualizes the results as a 3D brain with real-time region activation scores.

Built on Meta's TRIBE v2 foundation model and the HCP-MMP1 brain atlas.

# Setup Guide:

## Getting a GPU

TRIBE v2 requires a CUDA-capable GPU. [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) is one way to get one:

1. Create an account at [lambdalabs.com](https://lambdalabs.com)
2. Go to **Instances** → **Launch Instance**
3. Pick any GPU type (a single A10 or A100 works fine)
4. Add your SSH key (or create one in the Lambda dashboard under **SSH Keys**)
5. Launch the instance and wait for it to show **Running** under status
6. Copy the lambda IP address from the dashboard and open the terminal to connect:
   ```bash
   ssh -L 5173:localhost:5173 -L 8000:localhost:8000 -i ~/.ssh/<your_SSH_key.pem ubuntu@<your_lambda_instance_ip_address>
   ```

Once you're SSH'd into the instance, continue with setup below.

## Instance Setup

```bash
# Clone and enter the repo
git clone https://github.com/markuel/neuroLoop.git
cd neuroLoop

# Configure
cp .env.example .env
nano .env
# add tour hugging face token with access to Llama
#   HF_TOKEN=your_huggingface_token

# Run setup (installs everything + starts servers)
bash setup.sh
```

`setup.sh` handles everything automatically:

Once running, open **http://localhost:5173** in your browser.

## Usage

1. Click **Upload Video** or **Paste Text** in the top bar
2. Wait for TRIBE v2 to process (progress bar shows status)
3. View results:
   - **Left panel**: Video playback (or text display)
   - **Right panel**: 3D brain — rotate and zoom with mouse
   - **Timeline**: Scrub through time, play/pause
   - **Bottom panel**: Top 10 activated brain regions at the current timestep, color-coded by functional network

## Architecture

```
Browser (React + Three.js)  ←→  FastAPI (Lambda Cloud GPU)  ←→  AWS S3
```

- **Frontend**: React 19, Vite, Three.js (react-three-fiber), Tailwind CSS, Zustand
- **Backend**: FastAPI serving TRIBE v2 inference + neuroLoop region analysis
- **Storage**: AWS S3 for uploaded media and prediction results

## Citation

This project is built on TRIBE v2. If you use this software, please cite:

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

TRIBE v2 is created by Meta Platforms, Inc. The neuroLoop SDK and dashboard are built on top of TRIBE v2 under the same license terms.
