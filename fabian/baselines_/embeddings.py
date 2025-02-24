import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import os
import pickle  # For saving metadata if needed

DEFAULT_IMPORTANT_TERMS = {
    'service', 'quality', 'customer', 'product', 'shipping',
    'recommend', 'happy', 'great', 'good', 'bad', 'terrible',
    'excellent', 'awful', 'amazing', 'horrible', 'best', 'worst',
    'love', 'hate', 'helpful', 'useless', 'complaint', 'thank'
}
DEFAULT_NGRAM_RANGE = (2, 5)
DEFAULT_TOP_N = 20

@dataclass
class TextProcessorConfig:
    """Configuration for text processing (GPU part)."""
    semantic_model_name: str = 'all-MiniLM-L6-v2'  # Default semantic model
    output_embeddings_dir: str = 'output_embeddings' # Directory to save embeddings


class GPUEmbeddingGenerator:
    def __init__(self, config: Optional[TextProcessorConfig] = None):
        """Initialize GPUEmbeddingGenerator for GPU-based embedding."""
        self.config = config or TextProcessorConfig()
        self.semantic_model = SentenceTransformer(self.config.semantic_model_name) if self.config.semantic_model_name else None # Load semantic model on GPU
        if not os.path.exists(self.config.output_embeddings_dir):
            os.makedirs(self.config.output_embeddings_dir) # Create output directory if it doesn't exist

    def generate_and_save_embeddings_by_rating(self, df: pd.DataFrame, text_column: str, rating_column: str):
        """
        Generates sentence embeddings for texts grouped by rating and saves them to .npy files.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with text and rating columns.
        text_column : str
            Name of the text column.
        rating_column : str
            Name of the rating column.
        """
        if text_column not in df.columns or rating_column not in df.columns:
            raise ValueError(f"Columns {text_column} or {rating_column} not found in DataFrame")

        if self.semantic_model is None:
            raise ValueError("Semantic model is not initialized.")

        grouped = df.groupby(rating_column)[text_column]

        for rating, texts in grouped:
            text_list = texts.tolist()
            if not text_list:
                print(f"No texts found for rating {rating}. Skipping.")
                continue

            print(f"Generating embeddings for rating {rating}...")
            embeddings = self.semantic_model.encode(text_list, convert_to_numpy=True) # Generate embeddings on GPU

            output_file_path = os.path.join(self.config.output_embeddings_dir, f'rating_{rating}_embeddings.npy')
            np.save(output_file_path, embeddings) # Save embeddings to .npy file
            print(f"Embeddings for rating {rating} saved to: {output_file_path}")

            # Optionally save text list for reference (metadata - can be useful for debugging)
            metadata_file_path = os.path.join(self.config.output_embeddings_dir, f'rating_{rating}_texts.pkl')
            with open(metadata_file_path, 'wb') as f:
                pickle.dump(text_list, f) # Save original texts for reference


# --- Example Usage (GPU Script) ---
