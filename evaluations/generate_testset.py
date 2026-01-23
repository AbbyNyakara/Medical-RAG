from src.etl_pipeline.extractor import DocumentOCRExtractor
from src.etl_pipeline.chunker import DocumentChunkingPipeline, ChunkingConfig
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import json
from ragas.run_config import RunConfig

load_dotenv()

# Use the fetch_extracted_text from the Chunker (Fetches raw extracted text from s3)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestsetFromETL:
    """Generate test set from ETL pipeline"""

    def __init__(self, s3_bucket, region: str = 'us-east-1', chunking_config: ChunkingConfig = None):
        self.s3_bucket = s3_bucket
        self.region = region
        self.chunking_config = chunking_config or ChunkingConfig()

        # Initialize the ETL Components
        self.extractor = DocumentOCRExtractor(
            bucket=self.s3_bucket, region=self.region)
        self.chunker = DocumentChunkingPipeline(config=self.chunking_config)

    def load_extracted_text(self, s3_key: str):
        """Loads extracted text from s3 bucket"""
        try:
            logger.info("Loading text...")
            text = self.chunker.fetch_extracted_text(s3_key)
            return text
        except Exception as e:
            logger.error("Error returning text %s", e)
            raise

    def text_to_langchain_docs(self, text: str, source: str = "medical_document") -> List[Document]:
        """Convert Extracted text into Langchain documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )

        chunks = text_splitter.split_text(text)

        docs: List[Document] = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata={"source": source, "chunk_id": i}
                    )
                )
        logger.info("Created %d LangChain documents", len(docs))
        return docs

    def generate_synthetic_testset(self, s3_extracted_key: str, test_size: int = 10, source_name: str = "medical_document"):
        """Generate Synthetic test questions from s3 extracted docs"""
        generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # query_distribution = [
        #     (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)
        # ]
        # Use default configuration instead

        run_config = RunConfig(
        timeout=300,      
        max_retries=8,     
        max_workers=1,     
        max_wait=60,       
        log_tenacity=True  
        )


        try:
            # Load the text
            text = self.load_extracted_text(s3_extracted_key)
            # Create the docs
            documents = self.text_to_langchain_docs(text, source_name)
            if not documents:
                raise Exception("No documents created from extracted text")
            # Generate Synthetic text
            logger.info("\nGenerating synthetic test questions.")
            generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4-turbo", temperature=0), run_config=run_config)
            generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings
            )

            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=5, 
                #query_distribution=query_distribution # This will cause an increase in API costs. 
            )

            logger.info("Generated test set samples")
            return testset
        except Exception as e:
            logger.error("Failed to generate testset %s", e)
            raise

    def testset_to_questions(self, testset) -> list:
        """
        Convert RAGAS testset to question format
        Args:
            testset: Generated RAGAS testset
        Returns:
            List of {'question': str, 'reference': str}
        """
        questions = []
        
        for sample in testset.samples:
            questions.append({
                "question": sample.eval_sample.user_input,
                "reference": sample.eval_sample.reference,
                "contexts": sample.eval_sample.retrieved_contexts
            })
        
        logger.info("Converted %d samples to question format", len(questions))
        return questions
    
    def save_questions(self, questions: list, output_path: str):
        """Save generated questions to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(questions, f, indent=2)
        
        logger.info("Saved evaluation questions to %s", output_path)

    

    def process_document(self, file_path: str, test_size: int = 5, source_name: str = "medical_document"): # This is my local file
        """Complete pipeline in one call"""
        extraction_result = self.extractor.process_document(file_path)

        extracted_s3_key = extraction_result["saved_text_to"]
        testset = self.generate_synthetic_testset(
                s3_extracted_key=extracted_s3_key,
                test_size=test_size,
                source_name=source_name
            )
        questions = self.testset_to_questions(testset)

        return {
                "extracted_s3_key": extracted_s3_key,
                "testset": testset,
                "questions": questions,
            }
    
if __name__ == "__main__":
    testset_generator = TestsetFromETL(
        s3_bucket="medical-rag-docs-abigael-2026",
        region="us-east-1"
    )

    result = testset_generator.process_document(
        file_path="/Users/abigaelmogusu/projects/Medical-RAG/data/fake-aps.pdf",
        test_size=5,
        source_name="fake-aps.pdf"
    )

    # Access pieces if needed
    s3_key_extracted = result["extracted_s3_key"]
    questions = result["questions"]

    testset_generator.save_questions(
        questions=questions,
        output_path="evaluation_test_questions.json"
    )

    print("Extracted text S3 key:", s3_key_extracted)
    print("Generated", len(questions), "questions.")
