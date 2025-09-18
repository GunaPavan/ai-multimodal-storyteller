# src/ai_multimodal_storyteller/pipeline.py
import os
import torch
from ai_multimodal_storyteller.text_gen import StoryGenerator
from ai_multimodal_storyteller.image_gen import ImageGenerator
from ai_multimodal_storyteller.audio_gen import AudioGenerator
from ai_multimodal_storyteller.db import StoryDatabase

class StoryPipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.text_gen = StoryGenerator(device=self.device)
        self.image_gen = ImageGenerator(device=self.device)
        self.audio_gen = AudioGenerator()
        self.db = StoryDatabase()
        self.output_dir = "output"

    def run(self, prompt: str):
        os.makedirs(self.output_dir, exist_ok=True)
        images_dir = os.path.join(self.output_dir, "images")
        audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # 1️⃣ Generate story text
        story_text = self.text_gen.generate_story(prompt)
        scenes = self.text_gen.split_into_scenes(story_text)
        print(f"Generated story with {len(scenes)} scenes.")

        # 2️⃣ Process each scene
        for idx, scene in enumerate(scenes, 1):
            print(f"\nProcessing scene {idx}: {scene}")

            # Generate image with memory management
            image_path = os.path.join(images_dir, f"scene_{idx}.png")
            with torch.no_grad():
                self.image_gen.generate_image(scene, image_path)
            torch.cuda.empty_cache()  # free GPU memory after each image

            # Generate audio
            audio_path = os.path.join(audio_dir, f"scene_{idx}.wav")
            self.audio_gen.generate_audio(scene, audio_path)

            # Store in DB
            scene_id = f"scene_{idx}"
            self.db.add_scene(scene_id, scene, image_path, audio_path)

        print("\n✅ All scenes processed and stored successfully!")
        return scenes
