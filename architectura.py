# ARCHITECTURA: AI-Assisted Architectural Visualization Tool

# IMPORTS
# For Generating Image
import torch
import clip
import gradio as gr

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw


# Helper function for device fallback
def get_device():
    """Returns 'cuda' if a GPU is available, else defaults to 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# Load the models and pipelines
def setup_pipelines():
  """
  Loads and return all required AI models:
  - Stable Diffusion Inpainting Pipeline (image editing based on prompt)
  - CLIP model (text-image alignment scoring)
  """
  try:
      device = get_device()
      if device == "cuda":
          torch_dtype = torch.float16
      else:
          torch_dtype = torch.float32
      print(f"Using device: {device}")

      inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
          "runwayml/stable-diffusion-inpainting", torch_dtype=torch_dtype
      ).to(device)

      clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

      return inpaint_pipe, clip_model, clip_preprocess, device

  except Exception as e:
      print(f"Error loading models: {e}")
      raise


# Evaluation Metrics
def calculate_clip_score(clip_model, clip_preprocess, image, prompt, device):
    """
    Calculates Contrastive Language-Image Pretraining (CLIP) score between generated image and text prompt.
    
    CLIP Score range and interpretation:
        1.00  : Perfect alignment between image and text prompt.
        0.00  : No meaningful correlation.
        -1.00 : Indicates strong misalignment (rare and often indicative of bugs).
    """
    try:
        processed_image = clip_preprocess(image).unsqueeze(0).to(device)
        tokenized_prompt = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(processed_image)
            text_features = clip_model.encode_text(tokenized_prompt)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).item()
    
    except Exception as e:
        raise RuntimeError(f"Error calculating CLIP score: {e}")


# Core function
def generate_image_and_calculate_metrics(inpaint_pipe, clip_model, clip_preprocess, device, prompt, image):
    """
    Complete generation and evaluation pipeline:
    1. Download location image from URL.
    2. Generate base image from text prompt.
    3. Create mask for inpainting.
    4. Inpaint image using location and mask.
    5. Evaluate the result using CLIP scores.
    """
    try:
        # Step 1: Accept and preprocess uploaded location image
        location_image = image.convert("RGB").resize((512, 512))

        # Step 2: Create mask for inpainting
        mask_image = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.rectangle((150, 150, 350, 350), fill=255)

        # Step 3: Inpaint image using location and mask
        result_image = inpaint_pipe(
            prompt=prompt,
            image=location_image,
            mask_image=mask_image
        ).images[0]
            
        clip_score = calculate_clip_score(clip_model, clip_preprocess, result_image, prompt, device)

        return result_image, clip_score
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None, None, None


# Gradio Interface
def gradio_wrapper(prompt, image):
    """ Handles user input from Gradio UI (e.g., text, uploaded image) and routes it through the core function."""
    try:
        result_image, clip_score = generate_image_and_calculate_metrics(
            inpaint_pipe, clip_model, clip_preprocess, device, prompt, image
        )
        if result_image is None:
            return Image.new("RGB", (512, 512), (255, 0, 0)), "Error"

        return result_image, f"{clip_score:.2f}"

    except Exception as e:
        print(f"Error in Gradio wrapper: {e}")
        return Image.new("RGB", (512, 512), (255, 0, 0)), "Error"


# Load pipelines and models
inpaint_pipe, clip_model, clip_preprocess, device = setup_pipelines()


# Launch the App
app = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Textbox(label="Enter your design prompt", placeholder="e.g., a modern 2-story house with solar panels"),
        gr.Image(type="pil", label="Upload a location image")
    ],
    outputs=[
        "image",
        gr.Textbox(label="CLIP Score (closer to 1.0 = the image is more aligned with the text prompt)", placeholder="Range: -1.0 to 1.0"),
    ],
    title="Architectura - AI-Assisted Architectural Visualization Tool",
    description=(
        "Enter a design prompt and upload a location image."
        "AI will generate a design based on the location and prompt context, "
        "then evaluate it using CLIP score (prompt alignment)."
    )
)

# Launch the Gradio App
if __name__ == "__main__":
    app.launch(share=True)