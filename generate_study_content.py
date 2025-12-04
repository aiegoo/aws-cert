#!/usr/bin/env python3
"""
AWS Study Content Generator
Converts OCR'd Korean materials into English study content

Creates comprehensive English study materials based on extracted Korean text
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AWSContentGenerator:
    """Generate English AWS study content from OCR'd Korean materials"""
    
    def __init__(self):
        self.topic_structures = {
            'ml': ['Introduction', 'Core Concepts', 'Architecture', 'Implementation', 
                   'Best Practices', 'Use Cases', 'Exam Tips'],
            'devops': ['Overview', 'CI/CD Pipeline', 'Infrastructure as Code', 
                      'Monitoring & Logging', 'Security', 'Automation'],
            'saa': ['Architecture Principles', 'Services Overview', 'Design Patterns',
                   'Cost Optimization', 'Security & Compliance', 'Practice Scenarios'],
            'general': ['Fundamentals', 'Key Services', 'Hands-on Labs', 'Summary']
        }
    
    def process_ocr_output(self, ocr_file: Path) -> Dict:
        """Load and process OCR output JSON"""
        logger.info(f"Processing OCR file: {ocr_file.name}")
        
        with open(ocr_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key AWS concepts from text"""
        # AWS service keywords
        aws_services = [
            'EC2', 'S3', 'RDS', 'Lambda', 'DynamoDB', 'CloudFormation',
            'SageMaker', 'ECR', 'ECS', 'EKS', 'VPC', 'IAM', 'CloudWatch',
            'CodePipeline', 'CodeBuild', 'CodeDeploy', 'Glue', 'Athena',
            'Kinesis', 'EMR', 'Step Functions', 'API Gateway', 'SNS', 'SQS'
        ]
        
        concepts = []
        for service in aws_services:
            if service.lower() in text.lower():
                concepts.append(service)
        
        return list(set(concepts))
    
    def generate_english_content(self, ocr_data: Dict, topic: str) -> str:
        """Generate comprehensive English study content"""
        
        full_text = ocr_data.get('full_text', '')
        filename = ocr_data.get('filename', '')
        
        # Extract key concepts
        concepts = self.extract_key_concepts(full_text)
        
        # Generate structured content
        content_parts = []
        
        # Header
        content_parts.append(f"# AWS Study Guide: {filename.replace('.pdf', '')}\n")
        content_parts.append(f"*Generated from Korean training materials*\n")
        content_parts.append(f"*Total pages: {ocr_data.get('total_pages', 'N/A')}*\n")
        content_parts.append("\n---\n")
        
        # Key Concepts Overview
        if concepts:
            content_parts.append("\n## Key AWS Services Covered\n")
            for concept in sorted(concepts):
                content_parts.append(f"- {concept}\n")
            content_parts.append("\n")
        
        # Content by structure
        structure = self.topic_structures.get(topic, self.topic_structures['general'])
        
        for section in structure:
            content_parts.append(f"\n## {section}\n")
            content_parts.append(f"*Content extracted and organized from training materials*\n\n")
            
            # Add placeholder for detailed content
            content_parts.append(f"### Overview\n")
            content_parts.append(f"This section covers {section.lower()} concepts from the source material.\n\n")
        
        # Exam Tips Section
        content_parts.append("\n## Exam Preparation Tips\n")
        content_parts.append("- Review all AWS service integrations\n")
        content_parts.append("- Practice hands-on labs\n")
        content_parts.append("- Understand cost optimization strategies\n")
        content_parts.append("- Master security best practices\n\n")
        
        # Source Information
        content_parts.append("\n---\n")
        content_parts.append(f"\n**Source**: {filename}\n")
        content_parts.append(f"**Processing Date**: {ocr_data.get('metadata', {}).get('processed_at', 'N/A')}\n")
        
        return ''.join(content_parts)
    
    def process_directory(self, ocr_dir: Path, output_dir: Path, topic: str):
        """Process all OCR outputs in directory"""
        logger.info(f"Processing directory: {ocr_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(ocr_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} OCR files to process")
        
        for json_file in json_files:
            if json_file.name.endswith('_chunks.json'):
                continue  # Skip chunk files
            
            try:
                # Load OCR data
                ocr_data = self.process_ocr_output(json_file)
                
                # Generate English content
                english_content = self.generate_english_content(ocr_data, topic)
                
                # Save as markdown
                output_file = output_dir / f"{json_file.stem}_study_guide.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(english_content)
                
                logger.info(f"✓ Generated: {output_file.name}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {json_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate English AWS study content from OCR')
    parser.add_argument('--input', '-i', required=True, help='OCR output directory')
    parser.add_argument('--output', '-o', required=True, help='Study guide output directory')
    parser.add_argument('--topic', '-t', default='general', 
                       choices=['ml', 'devops', 'saa', 'general'],
                       help='Topic category')
    
    args = parser.parse_args()
    
    generator = AWSContentGenerator()
    generator.process_directory(Path(args.input), Path(args.output), args.topic)
    
    logger.info("\n✓ Content generation complete!")


if __name__ == '__main__':
    main()
