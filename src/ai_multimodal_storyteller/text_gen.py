# src/ai_multimodal_storyteller/text_gen.py
import os
import re
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class StoryGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Initialize LLM for story generation.
        Automatically uses GPU if available, otherwise falls back to CPU.
        """
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate device mapping
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True if self.device == "cpu" else False,
        )
        
        # Create pipeline with appropriate device
        device_id = 0 if self.device == "cuda" else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id,
        )
        
        print(f"Story generator loaded successfully on {self.get_device_info()}")

    def generate_story(self, prompt: str, max_length: int = 400) -> str:
        """
        Generate a story from user prompt.
        """
        print(f"Generating story with prompt: '{prompt[:50]}...'")
        
        try:
            output = self.generator(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = output[0]["generated_text"]
            print("Story generation completed successfully")
            return generated_text
            
        except Exception as e:
            print(f"Error during story generation: {e}")
            # Fallback: return the prompt as story if generation fails
            return prompt

    def split_into_scenes(self, story: str, sentences_per_scene: int = 2) -> list[str]:
        """
        Split story into short scenes.
        """
        # Clean up the story text
        story = re.sub(r'\s+', ' ', story).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', story)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group into scenes
        scenes = [
            " ".join(sentences[i:i + sentences_per_scene])
            for i in range(0, len(sentences), sentences_per_scene)
        ]
        
        # Filter out empty scenes
        return [s for s in scenes if s.strip()]

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
        if self.device == "cpu":
            # Enable model CPU offload if available
            if hasattr(self.model, 'enable_cpu_offload'):
                self.model.enable_cpu_offload()
            print("CPU memory optimization enabled")