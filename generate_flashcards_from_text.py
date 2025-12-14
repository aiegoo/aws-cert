#!/usr/bin/env python3
"""
Generate Domain-Specific Flashcards from Kindle OCR Text
Extracts key concepts and creates interactive HTML flashcards
"""

import re
import argparse
from pathlib import Path

def extract_concepts_for_domain(text, domain):
    """Extract key concepts based on domain"""
    concepts = []
    
    if "Data Preparation" in domain or "Data Ingestion" in domain:
        # Domain 1: Data Preparation & Ingestion
        patterns = [
            (r'(?:AWS\s+)?S3', 'Amazon S3'),
            (r'Kinesis\s+Data\s+Streams?', 'Amazon Kinesis Data Streams'),
            (r'Kinesis\s+Data\s+Firehose', 'Amazon Kinesis Data Firehose'),
            (r'AWS\s+Glue', 'AWS Glue'),
            (r'Data\s+Lake', 'Data Lake'),
            (r'Data\s+Warehouse', 'Data Warehouse'),
            (r'Lakehouse', 'Lakehouse'),
            (r'ETL\s+pipeline', 'ETL Pipeline'),
            (r'Schema-on-write', 'Schema-on-Write'),
            (r'Schema-on-read', 'Schema-on-Read'),
            (r'Three\s+V[\'"]?s', 'Three Vs (Volume, Velocity, Variety)'),
            (r'Redshift', 'Amazon Redshift'),
        ]
        
        for pattern, concept in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                concepts.append(concept)
    
    return list(set(concepts))[:30]  # Return up to 30 unique concepts

def generate_flashcard_html(concepts, domain, output_file):
    """Generate HTML flashcards"""
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AWS MLA-C01 Flashcards - {domain}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ text-align: center; color: white; margin-bottom: 20px; font-size: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .domain-header {{ background: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; text-align: center; }}
        .domain-tag {{ display: inline-block; background: #667eea; color: white; padding: 8px 20px; border-radius: 20px; font-weight: bold; }}
        .card {{ background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 20px; cursor: pointer; transition: transform 0.3s; }}
        .card:hover {{ transform: translateY(-5px); }}
        .card-inner {{ position: relative; width: 100%; min-height: 200px; text-align: center; transition: transform 0.6s; transform-style: preserve-3d; }}
        .card.flipped .card-inner {{ transform: rotateY(180deg); }}
        .card-front, .card-back {{ position: absolute; width: 100%; min-height: 200px; backface-visibility: hidden; display: flex; align-items: center; justify-content: center; padding: 30px; border-radius: 15px; }}
        .card-front {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .card-back {{ background: white; color: #333; transform: rotateY(180deg); border: 3px solid #667eea; }}
        .card-content {{ font-size: 1.3rem; font-weight: bold; }}
        .answer {{ font-size: 1.1rem; line-height: 1.8; }}
        .nav-buttons {{ display: flex; gap: 10px; justify-content: center; margin-top: 20px; flex-wrap: wrap; }}
        .nav-buttons button {{ background: white; color: #667eea; border: none; padding: 12px 25px; border-radius: 25px; cursor: pointer; font-weight: bold; transition: all 0.3s; font-size: 1rem; }}
        .nav-buttons button:hover {{ background: #667eea; color: white; transform: scale(1.05); }}
        .stats {{ text-align: center; color: white; margin: 20px 0; font-size: 1.1rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 AWS MLA-C01 Flashcards</h1>
        <div class="domain-header">
            <div class="domain-tag">{domain}</div>
            <p style="margin-top: 10px; color: #666;">Click any card to flip</p>
        </div>
        <div class="stats">{len(concepts)} Key Concepts</div>
'''
    
    # Generate cards for each concept
    for idx, concept in enumerate(concepts, 1):
        question, answer = generate_qa_for_concept(concept, domain)
        
        html += f'''
        <div class="card" onclick="this.classList.toggle('flipped')">
            <div class="card-inner">
                <div class="card-front">
                    <div class="card-content">{question}</div>
                </div>
                <div class="card-back">
                    <div class="answer">{answer}</div>
                </div>
            </div>
        </div>
'''
    
    html += '''
        <div class="nav-buttons">
            <button onclick="window.location.href='index.html'">← Back to Index</button>
            <button onclick="flipAll()">Flip All Cards</button>
            <button onclick="resetAll()">Reset All</button>
        </div>
    </div>

    <script>
        function flipAll() {
            document.querySelectorAll('.card').forEach(card => card.classList.add('flipped'));
        }
        function resetAll() {
            document.querySelectorAll('.card').forEach(card => card.classList.remove('flipped'));
        }
    </script>
</body>
</html>
'''
    
    Path(output_file).write_text(html, encoding='utf-8')
    print(f"Generated {len(concepts)} flashcards")
    print(f"Saved to: {output_file}")

def generate_qa_for_concept(concept, domain):
    """Generate Q&A for a specific concept"""
    
    qa_map = {
        "Amazon S3": (
            "What AWS service provides object storage with virtually unlimited scalability for data lakes?",
            "<strong>Amazon S3</strong><br><br>S3 is the foundation for data lakes, offering 11 9's of durability, lifecycle policies, versioning, and cost-effective storage classes (Standard, IA, Glacier). It supports batch and streaming ingestion."
        ),
        "Amazon Kinesis Data Streams": (
            "What AWS service captures and processes real-time streaming data with sub-second latency?",
            "<strong>Amazon Kinesis Data Streams</strong><br><br>Kinesis Data Streams provides real-time processing with custom applications. It retains data for 24 hours to 7 days, supports multiple consumers, and requires manual shard management."
        ),
        "Amazon Kinesis Data Firehose": (
            "What AWS service automatically loads streaming data into S3, Redshift, or Elasticsearch?",
            "<strong>Amazon Kinesis Data Firehose</strong><br><br>Firehose is a fully managed service that buffers and batches data before delivery. It handles transformations via Lambda and requires minimal operational overhead."
        ),
        "AWS Glue": (
            "What AWS service provides serverless ETL and data cataloging for data lakes?",
            "<strong>AWS Glue</strong><br><br>Glue offers a data catalog, crawlers for schema discovery, and ETL jobs using Spark. It integrates with S3, Redshift, and RDS for unified metadata management."
        ),
        "Data Lake": (
            "What architecture stores raw data in its native format using schema-on-read?",
            "<strong>Data Lake</strong><br><br>Data lakes use S3 for storage, support unstructured/semi-structured data, and defer schema definition until query time. They enable exploratory analytics and ML workloads."
        ),
        "Data Warehouse": (
            "What architecture optimizes structured data for BI and analytics using schema-on-write?",
            "<strong>Data Warehouse</strong><br><br>Data warehouses like Redshift require upfront ETL, use columnar storage, and optimize for complex SQL queries. They provide a single source of truth for reporting."
        ),
        "Lakehouse": (
            "What hybrid architecture combines data lake flexibility with data warehouse performance?",
            "<strong>Lakehouse</strong><br><br>Lakehouses use Delta Lake or Iceberg for ACID transactions on S3. They enable analytics and ML on the same platform with unified governance via Lake Formation."
        ),
        "ETL Pipeline": (
            "What process extracts, transforms, and loads data before storage?",
            "<strong>ETL Pipeline</strong><br><br>ETL pipelines transform data before loading into warehouses. AWS Glue provides serverless ETL with Python/Scala scripts and visual designers."
        ),
        "Schema-on-Write": (
            "What approach defines data structure before ingestion?",
            "<strong>Schema-on-Write</strong><br><br>Used by data warehouses, schema-on-write enforces structure during ingestion. This ensures data quality but reduces flexibility for exploratory analysis."
        ),
        "Schema-on-Read": (
            "What approach defers data structure definition until query time?",
            "<strong>Schema-on-Read</strong><br><br>Used by data lakes, schema-on-read stores raw data and applies structure during analysis. This enables flexibility but requires schema inference tools like Glue."
        ),
        "Three Vs (Volume, Velocity, Variety)": (
            "What three dimensions characterize big data challenges?",
            "<strong>Three Vs: Volume, Velocity, Variety</strong><br><br>Volume = data scale (GB to PB), Velocity = speed (batch vs streaming), Variety = data types (structured, semi-structured, unstructured)."
        ),
        "Amazon Redshift": (
            "What AWS data warehouse service provides petabyte-scale analytics?",
            "<strong>Amazon Redshift</strong><br><br>Redshift is a columnar MPP database optimized for OLAP. It supports Spectrum for querying S3 data and Concurrency Scaling for burst workloads."
        ),
    }
    
    return qa_map.get(concept, (
        f"What is {concept}?",
        f"<strong>{concept}</strong><br><br>A key AWS service or concept in the {domain} domain for machine learning workflows."
    ))

def main():
    parser = argparse.ArgumentParser(description='Generate flashcards from Kindle OCR text')
    parser.add_argument('--input', required=True, help='Input text file from OCR')
    parser.add_argument('--output', required=True, help='Output HTML file')
    parser.add_argument('--domain', default='Data Preparation for ML', help='Domain name')
    
    args = parser.parse_args()
    
    # Read OCR text
    text = Path(args.input).read_text(encoding='utf-8')
    
    # Extract concepts
    concepts = extract_concepts_for_domain(text, args.domain)
    
    if not concepts:
        print(f"No concepts found for domain: {args.domain}")
        return
    
    # Generate HTML
    generate_flashcard_html(concepts, args.domain, args.output)

if __name__ == '__main__':
    main()
