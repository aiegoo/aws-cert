# 🎯 AWS Certification Study Plan - Updated December 5, 2025

**Target**: MLA-C01 → SAA-C03 → DOP-C02

## MLA-C01 - Priority 1
**Focus**: Ml, Machine Learning, Sagemaker, Ai, Generative

### 📚 Materials to Process:
- [ ] **Generative AI Essentials (이론).pdf** (22.1MB) - *Split first*
- [ ] **ML Engineering on AWS.pdf** (46.1MB) - *Split first*
- [ ] **MLOps Engineering (이론).pdf** (44.0MB) - *Split first*
- [ ] AI Certifications_202510_1_1.pdf (8.1MB)
- [ ] AI Certifications_202510_2-1.pdf (3.7MB)
- [ ] **Developing Generative AI APP (이론).pdf** (42.1MB) - *Split first*
- [ ] AI Certifications_202510_1_3.pdf (3.7MB)

### 🎓 Recommended SkillBuilder Courses:
- [ ] Machine Learning Essentials
- [ ] Amazon SageMaker Training
- [ ] MLOps Implementation
- [ ] Generative AI with Amazon Bedrock

### 🛠️ Processing Commands:
```bash
# Split large file first
python3 pdf_splitter.py split "ai/Generative AI Essentials (이론).pdf" --output "chunks_Generative_AI_Essentials_(이론)"
python3 ocr_study_app.py process-dir "chunks_Generative_AI_Essentials_(이론)"
# Split large file first
python3 pdf_splitter.py split "mla/ML Engineering on AWS.pdf" --output "chunks_ML_Engineering_on_AWS"
python3 ocr_study_app.py process-dir "chunks_ML_Engineering_on_AWS"
# Split large file first
python3 pdf_splitter.py split "devops/MLOps Engineering (이론).pdf" --output "chunks_MLOps_Engineering_(이론)"
python3 ocr_study_app.py process-dir "chunks_MLOps_Engineering_(이론)"
python3 ocr_study_app.py process "general/AI Certifications_202510_1_1.pdf"
python3 ocr_study_app.py process "general/AI Certifications_202510_2-1.pdf"
# Split large file first
python3 pdf_splitter.py split "general/Developing Generative AI APP (이론).pdf" --output "chunks_Developing_Generative_AI_APP_(이론)"
python3 ocr_study_app.py process-dir "chunks_Developing_Generative_AI_APP_(이론)"
python3 ocr_study_app.py process "general/AI Certifications_202510_1_3.pdf"
```

## SAA-C03 - Priority 2
**Focus**: Architect, Solution, Design, Infrastructure

### 📚 Materials to Process:
- [ ] summary_Architecting on AWS_lastest.pdf (0.8MB)
- [ ] **Architecting on AWS 이론.pdf** (58.1MB) - *Split first*
- [ ] **Planning and Designing Databases 이론.pdf** (50.9MB) - *Split first*

### 🎓 Recommended SkillBuilder Courses:
- [ ] Architecting on AWS
- [ ] AWS Well-Architected Framework
- [ ] Solutions Architecture

### 🛠️ Processing Commands:
```bash
python3 ocr_study_app.py process "general/summary_Architecting on AWS_lastest.pdf"
# Split large file first
python3 pdf_splitter.py split "general/Architecting on AWS 이론.pdf" --output "chunks_Architecting_on_AWS_이론"
python3 ocr_study_app.py process-dir "chunks_Architecting_on_AWS_이론"
# Split large file first
python3 pdf_splitter.py split "general/Planning and Designing Databases 이론.pdf" --output "chunks_Planning_and_Designing_Databases_이론"
python3 ocr_study_app.py process-dir "chunks_Planning_and_Designing_Databases_이론"
```

## DOP-C02 - Priority 3
**Focus**: Devops, Deployment, Pipeline, Automation

### 📚 Materials to Process:
- [ ] **DevOps AWS (이론).pdf** (51.9MB) - *Split first*
- [ ] DevOps AWS (실습).pdf (14.3MB)
- [ ] DevOps AWS (강사필기).pdf (5.9MB)

### 🎓 Recommended SkillBuilder Courses:
- [ ] DevOps Engineering on AWS
- [ ] CI/CD Pipelines
- [ ] Infrastructure as Code

### 🛠️ Processing Commands:
```bash
# Split large file first
python3 pdf_splitter.py split "devops/DevOps AWS (이론).pdf" --output "chunks_DevOps_AWS_(이론)"
python3 ocr_study_app.py process-dir "chunks_DevOps_AWS_(이론)"
python3 ocr_study_app.py process "devops/DevOps AWS (실습).pdf"
python3 ocr_study_app.py process "devops/DevOps AWS (강사필기).pdf"
```

## General AWS Materials
### 📚 Supporting Materials:
- [ ] Developing on AWS (실습).pdf
- [ ] Build Modern Applications with NoSQL DB.pdf
- [ ] AWS Technical Essentials.pdf
- [ ] AWS Jam Event-guide.pdf
- [ ] Build Modern Applications with NoSQL DB (실습).pdf
- [ ] SAA 정리 노션.pdf
- [ ] Exam-SAA-2025-11-14.pdf
