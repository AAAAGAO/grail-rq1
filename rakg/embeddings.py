import os

import numpy as np


class Encoder:
    def __init__(self, model_name=None, batch_size=32):
        self.model_name = model_name or os.environ.get(
            "RAKG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.batch_size = batch_size
        self.model = None

    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
            )
        return self.model

    def _text(self, value, kind):
        return str(value or "")

    def encode(self, values, kind="document"):
        model = self._load()
        values = [self._text(value, kind) for value in values]
        return np.asarray(model.encode(
            values,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)

    def similarity(self, query, documents):
        query_vector = self.encode([query], "query")[0]
        document_vectors = self.encode(documents, "document")
        return document_vectors @ query_vector
