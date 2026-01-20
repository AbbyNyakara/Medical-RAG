"""
Document Chunking Pipeline with S3 Storage
Chunks extracted text and stores in S3
"""

import boto3
import uuid
import logging
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from pathlib import Path
import json
import sys
from dataclasses import dataclass


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from src.etl_pipeline.extractor import DocumentOCRExtractor


@dataclass
class ChunkingConfig:
    """Configuration for chunking and S3 storage"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    s3_bucket: str = "medical-rag-docs-abigael-2026"
    region: str = "us-east-1"


class DocumentChunkingPipeline:
    """
    Complete pipeline: Extract → Chunk → Store in S3
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)

    # ============ S3 Operations ============

    def fetch_extracted_text(self, s3_key: str) -> str:
        """Fetch extracted text from S3"""
        try:
            response = self.s3.get_object(Bucket=self.config.s3_bucket, Key=s3_key)
            text = response['Body'].read().decode('utf-8')
            logger.info("Fetched text from S3: %s", s3_key)
            return text
        except Exception as e:
            logger.error(f"Failed to fetch text: {e}")
            raise

    def save_chunks_to_s3(self, chunks: List[str], metadata_list: List[Dict], original_filename: str) -> str:
        """Save chunks and metadata to S3"""
        try:
            base_name = Path(original_filename).stem
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"chunks/{timestamp}/{unique_id}/{base_name}_chunks.json"

            chunks_data = {
                'document_name': original_filename,
                'processing_date': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'chunks': [
                    {'text': chunk, 'metadata': meta}
                    for chunk, meta in zip(chunks, metadata_list)
                ]
            }

            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=json.dumps(chunks_data, indent=2).encode('utf-8'),
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )
            logger.info(f"Saved chunks to S3: {s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            raise

    def fetch_chunks_from_s3(self, chunks_s3_key: str) -> List[Dict]:
        """Fetch chunks from S3 for vectorization"""
        try:
            response = self.s3.get_object(Bucket=self.config.s3_bucket, Key=chunks_s3_key)
            chunks_data = json.loads(response['Body'].read().decode('utf-8'))
            return chunks_data['chunks']
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            raise

    # ============ Chunking Operations ============

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using recursive character splitter"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                add_start_index=True
            )
            chunks = splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise

    def create_chunk_metadata(self, chunks: List[str], original_filename: str, text_s3_key: str, document_id: Optional[str] = None) -> tuple[List[Dict], str]:
        """Create metadata for each chunk"""
        if document_id is None:
            document_id = str(uuid.uuid4())

        metadata_list = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': f"{document_id}_chunk_{idx:04d}",
                'document_id': document_id,
                'original_filename': original_filename,
                'source_s3_key': text_s3_key,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk),
                'created_at': datetime.now().isoformat()
            }
            metadata_list.append(metadata)

        return metadata_list, document_id

    # ============ Main Pipeline ============

    def process_document(self, text_s3_key: str, original_filename: str, document_id: Optional[str] = None) -> Dict:
        """
        Complete pipeline: Fetch → Chunk → Store in S3
        
        Args:
            text_s3_key: S3 key of extracted text
            original_filename: Original document filename
            document_id: Optional document ID
        
        Returns:
            Processing result dictionary
        """
        try:
            logger.info("Starting processing: %s", original_filename)

            # Step 1: Fetch extracted text
            text = self.fetch_extracted_text(text_s3_key)

            # Step 2: Chunk text
            chunks = self.chunk_text(text)

            # Step 3: Create metadata
            chunk_metadata_list, doc_id = self.create_chunk_metadata(
                chunks, original_filename, text_s3_key, document_id
            )

            # Step 4: Save chunks to S3
            chunks_s3_key = self.save_chunks_to_s3(chunks, chunk_metadata_list, original_filename)

            result = {
                'success': True,
                'document_id': doc_id,
                'original_file': original_filename,
                'chunks_s3_key': chunks_s3_key,
                'total_chunks': len(chunks),
                'total_characters': sum(len(c) for c in chunks),
                'avg_chunk_size': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                'timestamp': datetime.now().isoformat()
            }

            logger.info("Processing complete: %d chunks created", len(chunks))
            return result

        except Exception as e:
            logger.error("Document processing failed: %s", e)
            return {
                'success': False,
                'error': str(e),
                'original_file': original_filename
            }
