# src/ai_multimodal_storyteller/db.py
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class StoryDatabase:
    def __init__(self, persist_dir: str = "output/db"):
        """
        Initialize ChromaDB client and embedding function.
        """
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create collection if it doesn't exist
        if "stories" not in [col.name for col in self.client.list_collections()]:
            self.collection = self.client.create_collection("stories")
        else:
            self.collection = self.client.get_collection("stories")

        # Embedding function using sentence-transformers
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def add_scene(self, scene_id: str, text: str, image_path: str, audio_path: str):
        """
        Add a single scene to the database.
        """
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[scene_id],
            metadatas=[{"text": text, "image": image_path, "audio": audio_path}],
            embeddings=[embedding],
        )

    def search_scene(self, query: str, n_results: int = 3):
        """
        Search scenes by text query.
        """
        query_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )
        return results
