# src/ai_multimodal_storyteller/app.py
import streamlit as st
import sys
from pathlib import Path

# Add project src folder to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from ai_multimodal_storyteller.pipeline import StoryPipeline
import os
from pathlib import Path

st.set_page_config(page_title="AI Multimodal Storyteller", layout="wide")

st.title("🎨 AI Multimodal Storyteller")
st.write(
    "Enter a prompt and generate a story with images and narration using open-source AI models."
)

# Prompt input
prompt = st.text_area("Enter your story prompt:", "")

# Device selection
device = st.selectbox("Select device:", ["cuda", "cpu"], index=0)

if st.button("Generate Story") and prompt.strip():
    with st.spinner("Generating story... This may take a few minutes depending on your device."):
        pipeline = StoryPipeline(device=device)
        scenes = pipeline.run(prompt)

    st.success("Story generation complete!")

    # Display results
    st.header("📖 Story Scenes")
    for idx, scene in enumerate(scenes, 1):
        st.subheader(f"Scene {idx}")
        st.write(scene)

        # Show image
        image_path = Path(f"output/images/scene_{idx}.png")
        if image_path.exists():
            st.image(str(image_path), caption=f"Scene {idx} Image", use_column_width=True)

        # Play audio
        audio_path = Path(f"output/audio/scene_{idx}.wav")
        if audio_path.exists():
            st.audio(str(audio_path))
