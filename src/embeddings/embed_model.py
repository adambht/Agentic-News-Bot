# src/embeddings/embed_model.py
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Path to optionally load a previously saved embedding model
SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/embedding_model")

# Load model: saved first, else pretrained
if os.path.exists(SAVED_MODEL_PATH):
    embed_model = SentenceTransformer(SAVED_MODEL_PATH)
else:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=HF_TOKEN)

def embed_text(df: pd.DataFrame, text_column: str = 'text') -> np.ndarray:
    """
    Minimal preprocessing + embeddings.
    """
    df['text_clean'] = df[text_column].astype(str).str.lower().str.strip()
    embeddings = embed_model.encode(df['text_clean'].tolist(), show_progress_bar=False)
    return embeddings

def preprocess_and_embed(df: pd.DataFrame, text_column: str = 'text') -> np.ndarray:
    """
    Wrapper to make a consistent interface for training/predicting.
    """
    return embed_text(df, text_column=text_column)
