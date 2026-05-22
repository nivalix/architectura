# Architectura
### AI-Assisted Architectural Visualization Tool

A proof-of-concept application that uses generative AI to produce architectural design concepts combined with real-world location images, evaluated using quantitative image quality metrics.

Originally built as a group project at BINUS University, then cleaned and improved for portofolio purposes.

---

## What It Does

Takes two inputs:
1. **Text prompt** describing an architectural concept (e.g. *"a modern 2-story house with balcony"*)
2. **Location image** of a real-world site

Runs them through an AI pipeline that consist of three stages:
| Stage | Model | Purpose |
|---|---|---|
| Inpainting | Stable Diffusion Inpainting | Generate and combine design into location |
| Alignment Check | CLIP (ViT-B/32) | Measure how well the output matches the prompt |

---

## Demo
| Input | Output |
|---|---|
| ![Input](demo/input.png) | ![Output](demo/output.png) |

> Prompt: *"Wooden cabin with two windows on each side"* with CLIP Score: 0.27

---

## Tech Stack
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [OpenAI CLIP (ViT-B/32)](https://github.com/openai/CLIP)
- [Gradio](https://gradio.app/) for interactive web UI
- PyTorch

---

## Setup

### Requirements
- Python 3.8+
- GPU recommended (CPU fallback supported but slow)

### Installation
```bash
git clone https://github.com/nivalix/architectura.git
cd architectura
pip install -r requirements.txt
```

### How to Run
```bash
python architectura.py
```

> If you're **Running on Colab**, Install dependencies with `!pip install -r requirements.txt` and run the script directly. Gradio will automatically generate a public link.

---

## Post-Submission Improvements

Changes made after academic submission to bring the code closer to production standards:

- Replaced URL-based image input with direct file upload (removes hotlinking issues)
- Removed unused `StableDiffusionPipeline` and dead code
- Removed unused imports (`requests`, `BytesIO`, `torchvision.transforms`)
- Removed FID score as it is considered a misapplied metric
- Restructured with proper docstrings, section comments, and clean function signatures
- Added `requirements.txt` and `.gitignore`

---

## Team
Built by a team of 3 CS students at BINUS University.
