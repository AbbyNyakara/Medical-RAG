import cohere
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from typing import List
import logging

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class RerankerConfig:
    """Configuration for reranker"""
    enable_reranking = True
    api_key = os.environ["COHERE_API_KEY"]
    similarity_threshold = 0.7
    model = "rerank-english-v3.0"
    top_k_rerank: int = 4


class SimpleReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.cohere_client = None

        if config.enable_reranking:
            self.cohere_client = cohere.ClientV2(api_key=config.api_key)
            logger.info("Initialized Cohere reranker")

        else:
            logger.info("Reranking is disabled")
            return 

    def rerank_results(self, query: str, documents: List[str], doc_ids: List[str]):
        """Rerank the documents for relevance to query"""
        try:
            response = self.cohere_client.rerank(
                model=self.config.model,
                query=query,
                documents=documents,
                top_n=self.config.top_k_rerank
            )

            reranked_ids = [doc_ids[r.index] for r in response.results]
            scores = [float(r.relevance_score) for r in response.results]

            logger.info("Reranked documents")
            return reranked_ids, scores
        except Exception as e:
            logger.error("Reranking Failed %s", e)
            return doc_ids, [1.0] * len(doc_ids)
