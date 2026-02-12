from ragas import EvaluationDataset, SingleTurnSample, evaluate
from src.rag.pipeline import MedicalRAGPipeline
from ragas.metrics.collections import (
    # Faithfulness,
    FactualCorrectness,
    AnswerRelevancy,
    SemanticSimilarity,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI

from typing import List
from dotenv import load_dotenv
import logging
import json
import pandas as pd

"""
# Faithfulness(llm=llm), -  how factually consistent a response is with the retrieved context
# FactualCorrectness(llm=llm), #extent to which the generated response aligns with the reference.
"""

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_questions(path: str) -> list:
    """Load evaluation questions JSON."""
    with open(path, "r") as file:
        return json.load(file)


def build_eval_dataset_from_file(path: str) -> EvaluationDataset:
    """
    Build initial EvaluationDataset from JSON file:
    """
    raw = load_questions(path)

    samples: List[SingleTurnSample] = []
    for row in raw:
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=row.get("contexts", []),
                response="",  # filled by rag pipeline
                reference=row.get("reference", ""),
            )
        )

    return EvaluationDataset(samples=samples)

def run_rag_on_dataset(
    dataset: EvaluationDataset,
    rag_pipeline: MedicalRAGPipeline,
) -> EvaluationDataset:
    """
    For each sample:
      - call MedicalRAGPipeline.answer_question(question)
      - fill in response and (optionally) updated contexts
    """
    filled_samples: List[SingleTurnSample] = []

    for sample in dataset.samples:
        question = sample.user_input
        rag_output = rag_pipeline.answer_question(question)

        answer = rag_output["answer"]
        contexts = rag_output.get("contexts", sample.retrieved_contexts)

        filled_samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=sample.reference,
            )
        )
        print(answer)

    return EvaluationDataset(samples=filled_samples)

def evaluate_medical_rag(questions_path: str):
    """
    Full evaluation:
      1) Load questions + references + seed contexts from JSON
      2) Run MedicalRAGPipeline to get responses + final contexts
      3) Evaluate with RAGAS metrics
    """
    base_dataset = build_eval_dataset_from_file(questions_path)

    rag_pipeline = MedicalRAGPipeline(
        s3_bucket="medical-rag-docs-abigael-2026"
    )
    eval_dataset = run_rag_on_dataset(base_dataset, rag_pipeline)

    client = AsyncOpenAI() 

    llm = llm_factory("gpt-4o-mini", client=client)
    emb = embedding_factory("openai",model="text-embedding-3-small", client=client)

    metrics = [
        
        AnswerRelevancy(llm=llm, embeddings=emb),
        SemanticSimilarity(embeddings=emb), #semantic resemblance between a generated response and a reference (ground truth) answer.
    ]

    # 🔍 DEBUG: print types and instances
    for i, m in enumerate(metrics):
        print(f"Metric {i}: {m}, type: {type(m)}")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    logger.info("Evaluation results:\n%s", result)
    return result


# ---------- Script entrypoint ----------

if __name__ == "__main__":
    questions_file = "evaluation_test_questions.json"
    scores = evaluate_medical_rag(questions_file)
    try:
        df = scores.to_pandas()
        df.to_csv("rag_evaluation_scores.csv", index=False)
        print(df)
    except AttributeError:
        print("Raw scores:", scores)
