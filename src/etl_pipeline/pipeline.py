
from pathlib import Path
import sys
from typing import Dict, List
import logging

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from src.etl_pipeline.extractor import DocumentOCRExtractor
from src.etl_pipeline.chunker import ChunkingConfig, DocumentChunkingPipeline
from src.etl_pipeline.embedder import EmbeddingPipeline, EmbeddingConfig, PineconeConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ETLPipeline:
    def __init__(self,
                 s3_bucket: str,
                 embedding_config: EmbeddingConfig = None,
                 pinecone_config: PineconeConfig = None,
                 chunking_config: ChunkingConfig = None,
                 region: str = "us-east-1"):

        self.s3_bucket = s3_bucket
        self.region = region

        self.extractor = DocumentOCRExtractor(bucket=s3_bucket, region=region)
        self.chunker = DocumentChunkingPipeline(
            chunking_config or ChunkingConfig())
        self.embedder = EmbeddingPipeline(
            embedding_config or EmbeddingConfig(),
            pinecone_config or PineconeConfig(),
            s3_bucket
        )

    def process_document(self, file_path: str):
        """
        Complete ETL pipeline in one call

        Args:
            file_path: Path to local PDF file -> change to uploaded via upload button and api

        Returns:
            Complete processing result
        """
        try:
            logger.info("Starting ETL pipeline: %s", file_path)

            # ============ STEP 1: EXTRACT ============
            logger.info("STEP 1: Extracting text from PDF...")
            extraction_result = self.extractor.process_document(file_path)

            if not extraction_result:
                raise Exception("Extraction failed")

            logger.info("Extracted characters")

            # ============ STEP 2: CHUNK ============
            logger.info("STEP 2: Chunking text...")
            chunking_result = self.chunker.process_document(
                text_s3_key=extraction_result['saved_text_to'],
                original_filename=extraction_result['original_file']
            )

            if not chunking_result['success']:
                raise Exception("Chunking failed: %s ",chunking_result['error'])

            logger.info("Chunks created")

            # ============ STEP 3: EMBED ============
            logger.info("STEP 3: Embedding chunks...")
            embedding_result = self.embedder.process_document(
                chunks_s3_key=chunking_result['chunks_s3_key']
            )

            if not embedding_result['success']:
                raise Exception(
                    f"Embedding failed: {embedding_result['error']}")

            logger.info(
                f"✓ Embedded: {embedding_result['stored']} vectors stored in Pinecone")

            # ============ COMPLETE RESULT ============
            result = {
                'success': True,
                'document_id': chunking_result['document_id'],
                'original_file': extraction_result['original_file'],
                'extraction': {
                    'uploaded_to': extraction_result['uploaded_to'],
                    'text_length': extraction_result['text_length'],
                    'saved_to': extraction_result['saved_text_to']
                },
                'chunking': {
                    'total_chunks': chunking_result['total_chunks'],
                    'total_characters': chunking_result['total_characters'],
                    'avg_chunk_size': chunking_result['avg_chunk_size'],
                    'chunks_s3_key': chunking_result['chunks_s3_key']
                },
                'embedding': {
                    'vectors_stored': embedding_result['stored'],
                    'timestamp': embedding_result['timestamp']
                }
            }

            logger.info("DOCUMENT PROCESSING COMPLETE")
            return result

        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_file': file_path
            }

# pipeline = ETLPipeline(s3_bucket="medical-rag-docs-abigael-2026")
# file = "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf"

# result = pipeline.process_document(file_path=file)
# print(result)
