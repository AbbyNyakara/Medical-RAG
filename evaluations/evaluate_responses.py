from ragas import EvaluationDataset, SingleTurnSample
from src.rag.pipeline import MedicalRAGPipeline
from typing import List
from dotenv import load_dotenv
import logging
import json

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_questions(path: str) -> list:
    """Load evaluation questions JSON."""
    with open(path, "r") as file:
        return json.load(file)


def run_rag_on_questions(questions_path: str, output_path: str = "rag_responses.json"):
    """
    Run RAG pipeline on questions and save responses to JSON.
    """
    raw_questions = load_questions(questions_path)
    rag_pipeline = MedicalRAGPipeline(
        s3_bucket="medical-rag-docs-abigael-2026"
    )
    results = []
    for item in raw_questions:
        question = item["question"]
        logger.info(f"Processing question: {question}")
        rag_output = rag_pipeline.answer_question(question)

        result = {
            "question": question,
            "answer": rag_output["answer"],
            "contexts": rag_output.get("contexts", []),
            "reference": item.get("reference", "")
        }
        
        results.append(result)
        print(f"\nQuestion: {question}")
        print(f"Answer: {rag_output['answer']}\n")
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved {len(results)} responses to {output_path}")
    return results

if __name__ == "__main__":
    questions_file = "evaluation_test_questions.json"
    output_file = "rag_responses.json"
    
    responses = run_rag_on_questions(questions_file, output_file)
    print(f"Results saved to: {output_file}")