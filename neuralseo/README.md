---
title: NeuralSEO
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: true
license: cc-by-nc-4.0
short_description: SEO tool powered by Meta AI TRIBE v2 brain encoder
hardware: zero-gpu
python_version: "3.12"
---

# 🧠 NeuralSEO

**The only SEO tool that tells you how the human brain actually responds to your content.**

Powered by [Meta AI's TRIBE v2](https://huggingface.co/facebook/tribev2) — a foundation model trained on 1,115 hours of fMRI recordings from 700+ human volunteers. TRIBE v2 predicts exactly how your brain responds to any piece of text. We weaponize that for SEO.

## Features

### 📝 Intro Paragraph Analyzer
Analyze only your opening paragraph (auto-trim to 600 chars) with TRIBE v2 neural signals.

- Paste intro text only (no URL scraping)
- TRIBE v2 scores hook strength, engagement, salience, and retention
- Gemini returns focused rewrite recommendations + improved intro draft
- Get a 0–100 intro neural score + dimension breakdown

### 📈 Neural CTR Predictor
Know your organic CTR before you publish. No A/B test. No guessing.

- Enter your keyword/topic
- Gemini generates dynamic title variants
- TRIBE v2 scores each by frontal attention network activation + salience
- Get a ranked list with neural engagement scores

## How It Works

TRIBE v2 outputs predicted fMRI activation across ~20,000 cortical vertices for any text input. We map those activations to SEO-relevant signals:

| Neural Signal | SEO Meaning |
|---|---|
| **Language comprehension activation** | Readability & clarity |
| **Frontal attention networks** | Will readers stay or bounce? |
| **Activation entropy (spatial complexity)** | E-E-A-T proxy — expert vs. thin content |
| **Salience network** | Does your title demand attention? |
| **Default Mode Network (inverse)** | Mind-wandering risk = bounce rate risk |

## Setup

### Environment Variables (set in HF Space Secrets)

```
GEMINI_API_KEY=your_gemini_api_key   # Required for recommendation generation
HF_TOKEN=your_huggingface_token      # Required for LLaMA 3.2 (gated model)
PARALLEL_API_KEY=your_parallel_key   # Optional if URL scraping is re-enabled
DATAFORSEO_LOGIN=your_login          # Optional: enriches SERP data
DATAFORSEO_PASSWORD=your_password
```

## License

Built on TRIBE v2 (CC BY-NC 4.0). NeuralSEO is free for non-commercial use.
