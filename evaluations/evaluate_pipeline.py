from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import (SingleHopSpecificQuerySynthesizer)
from ragas.metrics.collections import Faithfulness, FactualCorrectness, AnswerRelevancy, SemanticSimilarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict
import logging
from src.rag.pipeline import MedicalRAGPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





 