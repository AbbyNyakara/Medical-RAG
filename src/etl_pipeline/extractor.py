from pathlib import Path
from typing import Dict, List
import uuid
from datetime import datetime
import boto3
import time


class DocumentOCRExtractor:
    '''
    OCR Document Extraction using AWS Textract
    Handles:
      Upload document -> Extract text -> Store in S3  -> Clean up at end of session (After User is done)
    '''

    def __init__(self, bucket: str, region: str = 'us-east-1',):
        self.bucket = bucket
        self.region = region

        # Create AWS Clients
        self.textract = boto3.client('textract', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)

        # Track the files:
        self.uploaded_files: List[str] = []
        self.extracted_files: List[str] = []

    def upload_documents(self, file_path: str):
        '''
        # TODO - Perhaps this should also check the document type first? - To implement in API
        Uploads ducment to S3 bucket
        It creates a unique file name / file path using uuid and appends it to 
        the file name to create a s3 key locatioon to upload to
        '''
        file_name = Path(file_path).name
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"uploads/{timestamp}/{unique_id}/{file_name}"
        self.s3.upload_file(file_path, self.bucket, s3_key)

        # Add it to the list of uploaded files
        self.uploaded_files.append(s3_key)
        return s3_key

    def extract_text(self, s3_key: str):
        '''
        Performs OCR text extraction from the specified location within the s3 bucket (s3_key)
        '''
        response = self.textract.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': self.bucket,
                    'Name': s3_key,
                }
            },
            FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
        )

        job_id = response['JobId']
        extracted_text = self._wait_and_get_results(job_id)

        return extracted_text

    def _wait_and_get_results(self, job_id, max_attempts=60):
        '''
        Method checks: "Is the job done by textract?"
        Returns the extrcated string as a text
        '''
        for attempts in range(max_attempts):
            response = self.textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']

            if status == 'SUCCEEDED':
                return self._parse_textract_response(response)

            elif status == 'FAILED':
                error_msg = response.get('StatusMessage', 'Unknown error')
                raise Exception(f"Textract job failed: {error_msg}")

            elif status == 'IN_PROGRESS':
                time.sleep(5)

            else:
                print("Unexpected status")
                time.sleep(5)

        raise TimeoutError(
            f"Textract job did not complete within {max_attempts * 5} seconds")

    def _parse_textract_response(self, response: Dict):
        '''
        Parse Textract response and extract text.
        Handles pagination if results span multiple pages.
        Args:
            response: Textract API response

        Returns:
            Full extracted text
        '''
        all_text = []
        blocks = response.get('Blocks', [])

        for block in blocks:
            if block['BlockType'] == 'LINE':
                all_text.append(block.get('Text', ''))

        return '\n'.join(all_text)

    def save_extracted_text(self, text: str, original_filename: str) -> str:
        '''
        Save extracted text to S3 as a .txt file.

        Args:
            text: Extracted text to save
            original_filename: Original document filename

        Returns:
            S3 key where text was saved
        '''
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # gets file name w/o extention
        base_name = Path(original_filename).stem
        s3_key = f"extracted/{timestamp}_{unique_id}_{base_name}.txt"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=text.encode('utf-8'),
            ContentType='text/plain'
        )
        self.extracted_files.append(s3_key)
        return s3_key

    def process_document(self, file_path: str) -> Dict[str, str]:
        '''
        Complete workflow: Upload -> Extract -> Save.
        Args:
            file_path: Path to local document file    
        Returns:
            Dictionary with processing results
        '''
        filename = Path(file_path).name

        # Step 1: Upload document to S3
        upload_key = self.upload_documents(file_path)

        # Step 2: Extract text using Textract
        extracted_text = self.extract_text(upload_key)

        # Step 3: Save extracted text to S3
        text_key = self.save_extracted_text(extracted_text, filename)

        return {
            'original_file': filename,
            'uploaded_to': upload_key,
            'extracted_text': extracted_text,
            'saved_text_to': text_key,
            'text_length': len(extracted_text)
        }
