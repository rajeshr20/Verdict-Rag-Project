import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol defining the interface for embedding models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single query text."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        ...


class FakeEmbedder:
    """Deterministic, offline, hash-based embedder for testing and CI smoke runs.

    Generates bag-of-words hashed vectors normalized to unit length.
    Chunks with overlapping vocabulary with the query produce higher cosine similarity.
    Requires no GPU, no network, and no external API keys.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_vector(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._dimension

        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            # Deterministic bucket via sha256
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self._dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0

        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vector(text)


class SentenceTransformerEmbedder:
    """Production embedder using sentence-transformers (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer  # Lazy import
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return embeddings[0].tolist()


def get_embedder(model_name_or_type: str) -> Embedder:
    """Factory to instantiate the appropriate Embedder."""
    if model_name_or_type.lower() in ("fake", "mock", "test"):
        return FakeEmbedder()
    return SentenceTransformerEmbedder(model_name=model_name_or_type)
