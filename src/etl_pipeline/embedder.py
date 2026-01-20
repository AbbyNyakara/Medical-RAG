"""
Embedding pipeline: load chunks from S3 → embed text → store vectors + metadata in Pinecone
"""

from src.etl_pipeline.reranker import RerankerConfig, SimpleReranker
import boto3
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ========= Configs =========

@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    batch_size: int = 32


@dataclass
class PineconeConfig:
    """Configuration for Pinecone connection"""
    api_key: str = os.environ["PINECONE_API_KEY"]
    environment: str = "us-east-1"
    index_name: str = "medical-rag-index"
    metric: str = "cosine"
    dimension: int = 1536


# ========= Pipeline =========

class EmbeddingPipeline:
    def __init__(self,
                 embedding_config: EmbeddingConfig,
                 pinecone_config: PineconeConfig,
                 s3_bucket: str):

        self.config = embedding_config
        self.s3 = boto3.client("s3")
        self.bucket = s3_bucket

        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_config.model)

        # Pinecone
        self.pc = Pinecone(api_key=pinecone_config.api_key)
        self._setup_pinecone_index(pinecone_config)
        self.index = self.pc.Index(pinecone_config.index_name)

    def _setup_pinecone_index(self, config: PineconeConfig) -> None:
        """Create Pinecone index if it does not exist."""
        indexes = self.pc.list_indexes()
        if config.index_name not in [idx.name for idx in indexes.indexes]:
            self.pc.create_index(
                name=config.index_name,
                dimension=config.dimension,
                metric=config.metric,
                spec=ServerlessSpec(cloud="aws", region=config.environment),
            )
            logger.info("Created Pinecone index: %s", config.index_name)

    # ========= S3 I/O =========

    def load_chunks(self, s3_key: str) -> List[Dict]:
        """Load chunks JSON from S3: returns list of {'text': ..., 'metadata': {...}}."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            chunks = data.get("chunks", [])
            logger.info("Loaded %d chunks from %s", len(chunks), s3_key)
            return chunks
        except Exception as e:
            logger.error("Failed to load chunks: %s", e)
            raise

    # ========= Embedding =========

    def embed_chunks(self, chunks: List[Dict]) -> List[tuple]:
        """
        Generate embeddings for chunks.
        Returns: list of (chunk_id, embedding, metadata_with_text)
        """
        results: List[tuple] = []

        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i: i + self.config.batch_size]

            # 1) Extract plain text for embeddings
            texts = [c["text"] for c in batch]

            # 2) Call embeddings API
            try:
                embeddings = self.embeddings.embed_documents(texts)

                # 3) Attach IDs + metadata (include text for RAG)
                for chunk, emb in zip(batch, embeddings):
                    meta = chunk.get("metadata", {})
                    chunk_id = meta.get("chunk_id")

                    # Ensure we keep the text in metadata for retrieval
                    meta_with_text = {
                        **meta,
                        "text": chunk["text"],
                    }

                    results.append((chunk_id, emb, meta_with_text))

                logger.info(
                    "Embedded batch %d/%d",
                    i // self.config.batch_size + 1,
                    (len(chunks) + self.config.batch_size -
                     1) // self.config.batch_size,
                )

            except Exception as e:
                logger.error("Embedding failed: %s", e)
                raise

        return results

    # ========= Pinecone Storage =========

    def store_embeddings(self, vectors: List[tuple]) -> Dict:
        """
        Store embeddings in Pinecone.
        Input: [(id, embedding, metadata), ...]
        """
        try:
            pinecone_vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in vectors
            ]

            for i in range(0, len(pinecone_vectors), 100):
                batch = pinecone_vectors[i: i + 100]
                self.index.upsert(vectors=batch)

            logger.info("Stored %d vectors in Pinecone", len(vectors))
            return {"success": True, "count": len(vectors)}
        except Exception as e:
            logger.error("Storage failed: %s", e)
            raise

    # ========= Orchestration =========

    def process_document(self, chunks_s3_key: str) -> Dict:
        """Complete pipeline: Load → Embed → Store."""
        try:
            chunks = self.load_chunks(chunks_s3_key)
            vectors = self.embed_chunks(chunks)
            result = self.store_embeddings(vectors)

            return {
                "success": True,
                "chunks": len(chunks),
                "stored": result["count"],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            return {"success": False, "error": str(e)}

    # ========= Search (no rerank) =========

    def search_and_rerank(self, query: str, top_k: int = 10) -> Dict:
        """
        Embed user query → similarity search in Pinecone.
        Reranking can be added on top later.
        """
        try:
            query_embedding = self.embeddings.embed_query(query)

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )

            if not results.matches:
                return {"success": True, "results": []}

            final_results = []
            for match in results.matches:
                meta = match.metadata or {}
                final_results.append({
                    "id": match.id,
                    "score": float(match.score),
                    "text": meta.get("text", ""),
                    "source": meta.get("original_filename", "unknown"),
                })

            return {
                "success": True,
                "initial_results": len(results.matches),
                "final_results": len(final_results),
                "results": final_results,
            }

        except Exception as e:
            logger.error("Search failed: %s", e)
            raise
