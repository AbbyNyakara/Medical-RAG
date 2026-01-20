from dataclasses import dataclass
import os
import boto3


@dataclass
class AWSConfig:
    """AWS service configuration"""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    textract_role_arn: str = os.getenv("TEXTRACT_ROLE_ARN", "")

    def get_s3_client(self):
        return boto3.client('s3', region_name=self.region)

    def get_textract_client(self):
        return boto3.client('textract', region_name=self.region)
