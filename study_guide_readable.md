# AWS Machine Learning Engineering on AWS
## MLA-C01 Certification Study Guide

---

## Course Structure

This guide covers the AWS Machine Learning Engineer - Associate (MLA-C01) certification, focusing on building production ML systems on AWS. The course comprises 13 modules (0-12) covering the complete ML lifecycle: data preparation, model development, deployment, and MLOps.

**Prerequisites**: Basic ML concepts (supervised/unsupervised learning, training/validation splits), Python programming, AWS fundamentals (S3, EC2, IAM), and elementary statistics.

**Exam Domains**:
- Domain 1: Data Preparation for ML (28%)
- Domain 2: ML Model Development (26%)
- Domain 3: Deployment and Orchestration (22%)
- Domain 4: ML Solution Monitoring, Maintenance, and Security (24%)

---

## Module 0: AWS ML Services Architecture

### 0.1 Three-Layer ML Stack

AWS organizes ML services into three layers based on required ML expertise:

**Layer 1: AI Services** - Pre-trained models requiring no ML expertise. Services include Amazon Rekognition (computer vision), Comprehend (NLP), Polly (text-to-speech), Transcribe (speech-to-text), Translate (language translation), and Lex (conversational AI). These services handle model training, scaling, and maintenance automatically. Use cases: standard image recognition, sentiment analysis, language translation where custom training is unnecessary.

**Layer 2: ML Services** - Platform for custom ML with managed infrastructure. Amazon SageMaker serves as the primary service, providing end-to-end ML workflow management. Additional services include Amazon Forecast (time-series), Personalize (recommendations), and Fraud Detector (fraud detection). SageMaker components: Studio (IDE), Data Wrangler (data prep), Feature Store (feature management), Training (distributed training), Inference (deployment), Pipelines (workflow orchestration), Model Monitor (production monitoring).

**Layer 3: ML Frameworks & Infrastructure** - Full control over ML infrastructure. Includes Deep Learning AMIs (pre-configured EC2 instances), custom containers on ECS/EKS, AWS Inferentia/Trainium chips (custom ML hardware), and GPU instances (P3, P4, G4). Use when requiring fine-grained control over training infrastructure, custom distributed training strategies, or specialized hardware optimization.

### 0.2 ML Project Lifecycle

Production ML systems follow a seven-phase lifecycle: (1) Business problem definition - translate business objectives into ML tasks, (2) Data collection and preparation - gather, clean, validate training data, (3) Feature engineering - transform raw data into model-ready features, (4) Model training and evaluation - train models, tune hyperparameters, validate performance, (5) Model deployment - serve predictions at scale, (6) Monitoring and maintenance - track model performance, data drift, system health, (7) Model retraining - automated retraining when performance degrades or data distributions shift.

---

## Module 1: Machine Learning Fundamentals

### 1.1 Supervised Learning

Supervised learning trains models on labeled input-output pairs, enabling prediction on new inputs. Two primary types:

**Classification** - Predicts discrete categories. Binary classification assigns one of two labels (fraud/legitimate, churn/retain). Multi-class classification assigns one of N labels (product category, disease type). Multi-label classification assigns multiple labels per instance (article topics: politics AND economics). Common algorithms: Logistic Regression, Decision Trees, Random Forest, XGBoost, Neural Networks. AWS services: SageMaker XGBoost (tabular data), Linear Learner (linear relationships), Image Classification (ResNet CNN), BlazingText (text classification).

**Regression** - Predicts continuous numerical values. Use cases: price prediction, demand forecasting, temperature prediction, remaining useful life estimation. Common algorithms: Linear Regression, Ridge/Lasso Regression, Gradient Boosting Regression, Neural Networks. AWS services: SageMaker XGBoost, Linear Learner, DeepAR (time-series), Amazon Forecast (managed forecasting).

**Evaluation Metrics**:
- Classification: Accuracy = (TP + TN)/(TP + TN + FP + FN), Precision = TP/(TP + FP), Recall = TP/(TP + FN), F1 = 2 × (Precision × Recall)/(Precision + Recall), AUC-ROC
- Regression: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root MSE), R² (coefficient of determination), MAPE (Mean Absolute Percentage Error)

### 1.2 Unsupervised Learning

Unsupervised learning discovers patterns in unlabeled data without predefined outputs.

**Clustering** - Groups similar data points. K-Means partitions data into K clusters by minimizing within-cluster variance. Hierarchical clustering builds cluster dendrograms. DBSCAN identifies density-based clusters and outliers. Use cases: customer segmentation, document organization, anomaly detection. AWS service: SageMaker K-Means (web-scale clustering).

**Dimensionality Reduction** - Reduces feature space while preserving information. PCA (Principal Component Analysis) projects data onto orthogonal components explaining maximum variance. Use cases: data visualization, noise reduction, preprocessing for other algorithms. AWS service: SageMaker PCA.

**Anomaly Detection** - Identifies data points deviating from normal patterns. Random Cut Forest builds ensemble of random decision trees to compute anomaly scores. Use cases: fraud detection, equipment failure prediction, network intrusion detection. AWS services: SageMaker Random Cut Forest, Amazon Lookout services (for Metrics, Vision, Equipment).

### 1.3 Reinforcement Learning

Reinforcement learning trains agents to make sequential decisions by maximizing cumulative rewards. Agent interacts with environment, receives state observations, takes actions, and receives rewards/penalties. Policy defines action selection strategy. Value function estimates expected future rewards. Use cases: robotics control, resource optimization, game playing, autonomous vehicles. AWS services: SageMaker RL (integrated with Ray RLlib and Coach frameworks), AWS DeepRacer (autonomous racing simulation).

### 1.4 Amazon SageMaker Core Components

**SageMaker Studio** - Web-based IDE integrating entire ML workflow. Provides managed Jupyter notebooks, visual workflow builder, experiment tracking, model debugging, and team collaboration. Supports Python, R, and SQL kernels.

**SageMaker Data Wrangler** - Visual data preparation tool with 300+ built-in transformations. Features include data source integration (S3, Athena, Redshift, Snowflake), automated data quality insights, custom transformations via pandas/PySpark, and export to Feature Store or training jobs.

**SageMaker Feature Store** - Centralized repository for ML features with online and offline storage. Online store (DynamoDB-backed) provides low-latency feature retrieval for real-time inference (<10ms). Offline store (S3-based) supports batch processing and training. Supports feature versioning, point-in-time queries (prevents data leakage), and cross-team feature sharing.

**SageMaker Training** - Managed training infrastructure supporting distributed training, built-in algorithms (18+ pre-optimized algorithms), custom algorithms (bring your own container), spot instance training (up to 90% cost savings), and automatic model tuning (hyperparameter optimization).

**SageMaker Inference** - Deployment options include real-time endpoints (persistent HTTPS endpoints with auto-scaling), batch transform (asynchronous batch predictions), asynchronous inference (queued requests for large payloads), and serverless inference (pay-per-use, auto-scales to zero).

---

## Module 2: Framing ML Business Problems

### 2.1 Problem Definition Framework

Effective ML projects require translating business objectives into well-defined ML problems. Follow this framework:

**1. Business Objective Specification** - Define measurable business outcome (reduce customer churn by 15%, decrease fraud losses by $2M annually, improve delivery time prediction accuracy to 95%). Identify stakeholders, success criteria, and timeline.

**2. ML Problem Formulation** - Map business objective to ML task type. Classification: assign categories (will customer churn? is transaction fraudulent?). Regression: predict numerical values (expected delivery time, product demand). Clustering: discover segments (customer groups, product categories). Ranking: order items by relevance (search results, product recommendations).

**3. Data Requirements Assessment** - Determine required data: features (input variables), labels (for supervised learning), volume (typical: thousands to millions of examples), quality requirements (completeness, accuracy, consistency), and accessibility (storage location, access permissions, update frequency).

**4. Constraint Identification** - Technical constraints: latency requirements (real-time <100ms vs batch processing), cost budget (training and inference costs), infrastructure limitations (on-premises data, network bandwidth). Business constraints: interpretability requirements (GDPR "right to explanation", loan decision explanations), compliance (HIPAA, PCI-DSS, GDPR), risk tolerance (acceptable error rates).

**5. Success Metrics Definition** - Establish both business and technical metrics. Business metrics: ROI, revenue impact, cost savings, customer satisfaction, operational efficiency. Technical metrics: model accuracy, precision, recall, F1 score, AUC-ROC, MAE, RMSE. Define minimum acceptable performance thresholds.

### 2.2 ML Problem Type Selection

**Binary Classification** - Two possible outcomes. Examples: fraud detection (fraud/legitimate), churn prediction (churn/retain), loan approval (approve/deny), medical diagnosis (disease/healthy). Choose precision-focused metrics when false positives are costly. Choose recall-focused metrics when false negatives are costly.

**Multi-Class Classification** - Multiple mutually exclusive categories. Examples: product categorization (electronics/clothing/books), image classification (cat/dog/bird/fish), sentiment analysis (very negative/negative/neutral/positive/very positive). Use softmax activation for probability distribution across classes. Evaluation: confusion matrix, per-class precision/recall, macro/micro averaging.

**Regression** - Continuous numerical prediction. Examples: house price prediction, demand forecasting, temperature prediction, customer lifetime value estimation. Evaluation metrics: MAE (interpretable, same units as target), RMSE (penalizes large errors more), R² (proportion of variance explained), MAPE (percentage error).

**Clustering** - Unsupervised grouping. Examples: customer segmentation (identify distinct customer types), document clustering (organize similar documents), anomaly detection (identify outliers). Choose K (number of clusters) using elbow method, silhouette analysis, or business requirements. Evaluation: silhouette score, Davies-Bouldin index, within-cluster sum of squares.

**Time-Series Forecasting** - Predict future values from historical sequences. Examples: sales forecasting, demand prediction, stock price prediction, energy consumption forecasting. Handle seasonality (weekly, monthly, yearly patterns), trends (long-term increases/decreases), and external variables (holidays, promotions). AWS services: Amazon Forecast (automated forecasting), SageMaker DeepAR (RNN-based forecasting for related time series).

### 2.3 Feasibility Assessment

Before implementing ML solution, evaluate feasibility:

**Data Availability** - Sufficient labeled examples (supervised learning typically requires 1,000+ examples per class, deep learning requires 10,000+). Historical data covers relevant patterns. Labels are accurate and consistent. Data represents production distribution (no sampling bias).

**Pattern Existence** - Relationship exists between features and target. Simple exploratory analysis reveals correlations. Domain experts confirm predictive signals exist. Baseline models (simple rules, statistical methods) show performance above random.

**ML Appropriateness** - Problem cannot be solved adequately with rules-based system. Patterns are too complex for manual feature engineering. Data-driven approach outperforms expert-designed rules. Problem requires adaptation to changing patterns.

**Resource Availability** - Budget allocated for infrastructure, data collection, engineering effort. Technical expertise available (ML engineers, data scientists, MLOps engineers). Timeline realistic for data collection, model development, deployment, and validation. Management support for iterative development and potential initial failures.

**When NOT to Use ML** - Simple rules work perfectly (if temperature > 100°F, alert). Insufficient data (< 100 labeled examples). No pattern exists (purely random outcomes). Complete interpretability required (life-critical medical decisions). Problem changes faster than retraining cycle. Cost exceeds benefit (expensive ML infrastructure for minimal improvement).

---

*[Modules 3-12 to be written in same precise, technical style]*