# src/ai_multimodal_storyteller/audio_gen.py
import os
import torch
from TTS.api import TTS

class AudioGenerator:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """
        Initialize Coqui TTS model.
        Default: Tacotron2-DDC (high quality, small enough to run on CPU/GPU)
        Automatically uses GPU if available, otherwise falls back to CPU.
        """
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize TTS with the appropriate device
        self.tts = TTS(model_name).to(self.device)

    def generate_audio(self, text: str, output_path: str) -> str:
        """
        Generate speech from text and save as WAV.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate audio using the configured device
        self.tts.tts_to_file(text=text, file_path=output_path)
        
        print(f"Audio generated and saved to: {output_path}")
        return output_path

    def get_device_info(self) -> str:
        """
        Returns information about the current device being used.
        """
        if self.device == "cuda":
            return f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        else:
            return "CPU"