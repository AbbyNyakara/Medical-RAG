# from ragas import EvaluationDataset, SingleTurnSample, evaluate
# from src.rag.pipeline import MedicalRAGPipeline
# import logging
# from typing import List
# import json
# from ragas.metrics.collections import (
#     Faithfulness,
#     FactualCorrectness,
#     AnswerRelevancy,
#     SemanticSimilarity,
# )
# from openai import OpenAI
# from ragas.llms import llm_factory
# from ragas.embeddings import OpenAIEmbeddings
# import logging
# import pandas
# from dotenv import load_dotenv

# load_dotenv()


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# def load_questions(path):
#     with open(path, "r") as file:
#         return json.load(file)
    
# def build_evaluation_dataset(questions_path: str) -> EvaluationDataset:
#     raw = load_questions(questions_path)
#     samples: List[SingleTurnSample] = []
#     for q in raw:
#         samples.append(
#             SingleTurnSample(
#                 user_input=q["question"],
#                 retrieved_contexts=q.get("contexts", []),
#                 # response will be filled later by running the RAG pipeline
#                 response="",  
#                 reference=q.get("reference", ""),
#             )
#         )

#     dataset = EvaluationDataset(samples=samples)
#     return dataset

# def build_eval_dataset_from_file(path: str) -> EvaluationDataset:
#     with open(path, "r") as f:
#         raw = json.load(f)

#     samples: List[SingleTurnSample] = []
#     for row in raw:
#         samples.append(
#             SingleTurnSample(
#                 user_input=row["question"],
#                 retrieved_contexts=row.get("contexts", []),
#                 response="",                 
#                 reference=row.get("reference", ""),
#             )
#         )

#     return EvaluationDataset(samples=samples)

# def run_rag_on_dataset(dataset: EvaluationDataset,rag_pipeline: MedicalRAGPipeline) -> EvaluationDataset:

#     filled_samples: List[SingleTurnSample] = []
#     for sample in dataset.samples:
#         question = sample.user_input
#         rag_output = rag_pipeline.answer_question(question)

#         answer = rag_output['answer']
#         # contexts = rag_output.get("contexts", sample.retrieved_contexts)

#         filled_samples.append(
#             SingleTurnSample(
#                 user_input=question,
#                 # retrieved_contexts="", 
#                 response=answer, 
#                 reference=sample.reference
#             )
#         )

#     return EvaluationDataset(samples=filled_samples)


# def evaluate_medical_rag(questions_path: str):
#     base_dataset = build_eval_dataset_from_file("evaluation_test_questions.json")
#     rag_pipeline = MedicalRAGPipeline(s3_bucket="medical-rag-docs-abigael-2026")  
#     eval_dataset = run_rag_on_dataset(base_dataset, rag_pipeline)
#     client = OpenAI()
#     # Modern RAGAS LLM + embeddings
#     llm = llm_factory("gpt-4o-mini", client=client)
#     emb = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         client=client,
#     )

#     metrics = [
#         Faithfulness(llm=llm),
#         FactualCorrectness(llm=llm),
#         AnswerRelevancy(llm=llm, embeddings=emb),
#         SemanticSimilarity(embeddings=emb),
#     ]

#     result = evaluate(
#         dataset=eval_dataset,
#         metrics=metrics,
#     )

#     logger.info("Evaluation results:\n%s", result)
#     return result


# if __name__ == "__main__":
#     questions_file = "evaluation_test_questions.json"
#     scores = evaluate_medical_rag(questions_file)
#     df = scores.to_pandas()   
#     df.to_csv("rag_evaluation_scores.csv", index=False)
#     print(scores)


from ragas import EvaluationDataset, SingleTurnSample, evaluate
from src.rag.pipeline import MedicalRAGPipeline
from ragas.metrics.collections import (
    Faithfulness,
    FactualCorrectness,
    AnswerRelevancy,
    SemanticSimilarity,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

from typing import List
from dotenv import load_dotenv
import logging
import json
import pandas as pd

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- Data loading & dataset construction ----------

def load_questions(path: str) -> list:
    """Load evaluation questions JSON."""
    with open(path, "r") as file:
        return json.load(file)


def build_eval_dataset_from_file(path: str) -> EvaluationDataset:
    """
    Build initial EvaluationDataset from JSON file:
    [
      {"question": "...", "reference": "...", "contexts": ["...", ...]},
      ...
    ]
    """
    raw = load_questions(path)

    samples: List[SingleTurnSample] = []
    for row in raw:
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=row.get("contexts", []),
                response="",  # will be filled by the RAG pipeline
                reference=row.get("reference", ""),
            )
        )

    return EvaluationDataset(samples=samples)


# ---------- Run your RAG pipeline over all questions ----------

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
        # use contexts returned by pipeline if present, otherwise keep original
        contexts = rag_output.get("contexts", sample.retrieved_contexts)

        filled_samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=sample.reference,
            )
        )

    return EvaluationDataset(samples=filled_samples)


# ---------- Main evaluation entrypoint ----------

def evaluate_medical_rag(questions_path: str):
    """
    Full evaluation:
      1) Load questions + references + seed contexts from JSON
      2) Run MedicalRAGPipeline to get responses + final contexts
      3) Evaluate with RAGAS metrics
    """
    # 1. Build base dataset from file
    base_dataset = build_eval_dataset_from_file(questions_path)

    # 2. Initialize your RAG pipeline
    rag_pipeline = MedicalRAGPipeline(
        s3_bucket="medical-rag-docs-abigael-2026"
    )

    # 3. Run RAG over all questions
    eval_dataset = run_rag_on_dataset(base_dataset, rag_pipeline)

    # 4. Configure RAGAS LLM + embeddings (modern API)
    client = OpenAI()  # uses OPENAI_API_KEY from env

    llm = llm_factory("gpt-4o-mini", client=client)
    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        client=client,
    )

    metrics = [
        Faithfulness(llm=llm),
        FactualCorrectness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        SemanticSimilarity(embeddings=emb),
    ]

    # Optional sanity check
    for m in metrics:
        logger.info("Using metric: %s (%s)", m, type(m))

    # 5. Run evaluation
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

    # If your ragas version supports to_pandas()
    try:
        df = scores.to_pandas()
        df.to_csv("rag_evaluation_scores.csv", index=False)
        print(df)
    except AttributeError:
        # Fall back to printing raw scores
        print("Raw scores:", scores)
