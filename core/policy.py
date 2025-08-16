
import os
from typing import List, Dict
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class PolicyRAG:
    def __init__(self, persist_dir: str = '.chromadb'):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory=persist_dir))
        self.collection = self.client.get_or_create_collection(name='policies')
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.sa = SentimentIntensityAnalyzer()

    def _embed(self, texts: List[str]):
        return self.embedder.encode(texts).tolist()

    def add_pdf(self, path: str, meta: Dict=None):
        doc = fitz.open(path)
        docs, metadatas, ids = [], [], []
        for i, page in enumerate(doc):
            text = page.get_text('text')
            if text and text.strip():
                docs.append(text)
                metadatas.append({'page': i+1, 'source': os.path.basename(path), **(meta or {})})
                ids.append(f"{os.path.basename(path)}_{i+1}")
        if docs:
            embs = self._embed(docs)
            # upsert with unique ids
            self.collection.upsert(documents=docs, embeddings=embs, metadatas=metadatas, ids=ids)

    def query(self, q: str, k:int=5):
        q_emb = self._embed([q])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        out = []
        for doc, meta, dist in zip(res.get('documents',[[]])[0], res.get('metadatas',[[]])[0], res.get('distances',[[]])[0]):
            out.append({'text': doc, 'meta': meta, 'distance': dist})
        return out

    def sentiment(self, text: str):
        return self.sa.polarity_scores(text)
