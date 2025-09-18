# src/ai_multimodal_storyteller/image_gen.py
import os
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
import torch

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class ImageGenerator:
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize Stable Diffusion XL pipeline.
        Automatically uses GPU if available, otherwise falls back to CPU.
        """
        # Check if CUDA is available and set device accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Choose appropriate dtype based on device
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load the pipeline with appropriate settings
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN,
            torch_dtype=dtype,
        )
        
        # Move pipeline to the appropriate device
        self.pipe = self.pipe.to(self.device)
        
        # Optimize for CPU if needed
        if self.device == "cpu":
            print("Optimizing for CPU usage...")
            self.pipe.enable_attention_slicing()
        
        print(f"Model loaded successfully on {self.get_device_info()}")

    def generate_image(self, prompt: str, output_path: str) -> str:
        """
        Generate an image from a text prompt and save to file.
        """
        print(f"Generating image for prompt: '{prompt}'")
        
        # Generate image
        image = self.pipe(prompt).images[0]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return output_path

    def get_device_info(self) -> str:
        """
        Returns information about the current device being used.
        """
        if self.device == "cuda":
            return f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        else:
            return "CPU"

    def optimize_for_memory(self):
        """
        Additional optimizations for memory-constrained environments.
        """
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
        print("Memory optimization enabled")