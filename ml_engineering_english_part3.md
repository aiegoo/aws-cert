# AWS Machine Learning Engineering on AWS - Study Guide (Part 3)

> **Final Section: Modules 10-12**

---

## Module 10: MLOps and Automated Deployment

### Learning Objectives
- Understand MLOps principles and practices
- Implement automated ML pipelines
- Learn CI/CD for ML workflows
- Manage model lifecycle at scale

### Key Topics

#### 10.1 MLOps Fundamentals

**What is MLOps?**
Machine Learning Operations - practices that combine ML, DevOps, and Data Engineering to automate and scale ML workflows.

**MLOps Goals:**
- **Automation**: Reduce manual intervention
- **Reproducibility**: Same inputs → same outputs
- **Versioning**: Track models, data, code
- **Monitoring**: Detect issues early
- **Collaboration**: Teams work together efficiently
- **Governance**: Compliance and auditability

**MLOps Lifecycle:**
```
┌─────────────────────────────────────────────┐
│  Data → Feature Engineering → Training      │
│    ↓          ↓                    ↓        │
│  Validation → Model Registry → Deployment   │
│    ↓                              ↓        │
│  Monitoring ← ← ← ← ← ← ← ← ← ← ← ┘        │
│    ↓                                        │
│  Retrain (if drift detected)                │
└─────────────────────────────────────────────┘
```

**MLOps Maturity Levels:**

**Level 0: Manual Process**
- Manual training
- Manual deployment
- No versioning
- Infrequent updates

**Level 1: ML Pipeline Automation**
- Automated training
- Automated evaluation
- Manual deployment
- Versioning in place

**Level 2: CI/CD Pipeline Automation**
- Automated testing
- Automated deployment
- Automated monitoring
- Continuous training

#### 10.2 SageMaker Pipelines

**Purpose:**
- Orchestrate ML workflows
- Automate training and deployment
- Version control
- Reproducibility

**Pipeline Components:**

**1. Processing Step:**
```python
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    image_uri='my-processing-image',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1
)

processing_step = ProcessingStep(
    name='PreprocessData',
    processor=processor,
    inputs=[
        ProcessingInput(
            source='s3://bucket/raw-data',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination='s3://bucket/processed-data'
        )
    ],
    code='preprocessing.py'
)
```

**2. Training Step:**
```python
from sagemaker.workflow.steps import TrainingStep
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='training-image',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    hyperparameters={
        'epochs': 100,
        'learning_rate': 0.01
    }
)

training_step = TrainingStep(
    name='TrainModel',
    estimator=estimator,
    inputs={
        'training': TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri
        )
    }
)
```

**3. Evaluation Step:**
```python
from sagemaker.workflow.steps import ProcessingStep

evaluation_step = ProcessingStep(
    name='EvaluateModel',
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination='/opt/ml/processing/model'
        ),
        ProcessingInput(
            source='s3://bucket/test-data',
            destination='/opt/ml/processing/test'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='evaluation',
            source='/opt/ml/processing/evaluation'
        )
    ],
    code='evaluation.py'
)
```

**4. Conditional Step:**
```python
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Extract accuracy from evaluation
accuracy = JsonGet(
    step_name=evaluation_step.name,
    property_file='evaluation',
    json_path='metrics.accuracy'
)

# Condition: accuracy >= 0.85
condition = ConditionGreaterThanOrEqualTo(
    left=accuracy,
    right=0.85
)

condition_step = ConditionStep(
    name='CheckAccuracy',
    conditions=[condition],
    if_steps=[register_model_step],  # if True
    else_steps=[fail_step]  # if False
)
```

**5. Model Registration Step:**
```python
from sagemaker.workflow.step_collections import RegisterModel

register_step = RegisterModel(
    name='RegisterModel',
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=['text/csv'],
    response_types=['application/json'],
    inference_instances=['ml.m5.xlarge'],
    transform_instances=['ml.m5.xlarge'],
    model_package_group_name='my-model-group',
    approval_status='PendingManualApproval'
)
```

**6. Complete Pipeline:**
```python
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name='ml-pipeline',
    parameters=[
        instance_type_param,
        epochs_param,
        learning_rate_param
    ],
    steps=[
        processing_step,
        training_step,
        evaluation_step,
        condition_step
    ]
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

#### 10.3 Model Registry

**Purpose:**
- Centralize model storage
- Version management
- Model lineage
- Approval workflows
- Deployment tracking

**Model Package Groups:**
```python
from sagemaker.model import ModelPackageGroup

model_package_group = ModelPackageGroup(
    name='fraud-detection-models',
    description='Fraud detection model versions'
)
model_package_group.create()
```

**Registering Models:**
```python
from sagemaker.model import ModelPackage

model_package = ModelPackage(
    role=role,
    model_data='s3://bucket/model.tar.gz',
    model_package_group_name='fraud-detection-models',
    inference_instances=['ml.m5.xlarge'],
    transform_instances=['ml.m5.xlarge'],
    content_types=['text/csv'],
    response_types=['application/json'],
    approval_status='PendingManualApproval',
    model_metrics={
        'accuracy': 0.92,
        'precision': 0.88,
        'recall': 0.91
    }
)
model_package.register()
```

**Model Approval:**
```python
# List pending models
pending_models = ModelPackageGroup(name='fraud-detection-models').list_model_packages(
    approval_status='PendingManualApproval'
)

# Approve model
model_package = ModelPackage(
    model_package_arn='arn:aws:sagemaker:...'
)
model_package.update(approval_status='Approved')
```

**Model Lineage:**
```python
# Query lineage
from sagemaker.lineage.query import LineageQuery

query = LineageQuery(sagemaker_session)

# Find dataset used for model
query_filter = {
    'DestinationArn': model_package_arn,
    'LineageEntityType': 'Artifact'
}

lineage_results = query.query(query_filter=query_filter)
```

#### 10.4 CI/CD for ML

**Continuous Integration:**
- Automated testing
- Code quality checks
- Data validation
- Model testing

**Continuous Deployment:**
- Automated deployment
- Gradual rollout
- Rollback capability
- Production monitoring

**CI/CD Pipeline Architecture:**
```
Code Commit → Build → Test → Deploy Staging → Test Staging → Deploy Production
     ↓          ↓      ↓           ↓               ↓                ↓
  GitHub    CodeBuild  Unit    Model Test     Integration    Blue/Green
            ECR Build  Tests   Performance    Tests          Deployment
```

**AWS CodePipeline for ML:**

**Pipeline Definition:**
```yaml
version: 1
phases:
  pre_build:
    commands:
      - echo "Running data validation"
      - python validate_data.py
      
  build:
    commands:
      - echo "Building container"
      - docker build -t ml-training:latest .
      - docker tag ml-training:latest $ECR_REPO:latest
      - docker push $ECR_REPO:latest
      
  post_build:
    commands:
      - echo "Triggering SageMaker pipeline"
      - aws sagemaker start-pipeline-execution --pipeline-name ml-pipeline
```

**GitHub Actions Example:**
```yaml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
      - name: Validate data schema
        run: python validate_schema.py
        
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Start SageMaker Pipeline
        run: |
          aws sagemaker start-pipeline-execution \
            --pipeline-name ml-training-pipeline
```

#### 10.5 Infrastructure as Code

**AWS CloudFormation:**

**SageMaker Resources Template:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'SageMaker ML Infrastructure'

Parameters:
  ModelName:
    Type: String
    Default: 'fraud-detection-model'

Resources:
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        
  MLBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-ml-data'
      VersioningConfiguration:
        Status: Enabled
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
              
  ModelPackageGroup:
    Type: AWS::SageMaker::ModelPackageGroup
    Properties:
      ModelPackageGroupName: !Ref ModelName
      ModelPackageGroupDescription: 'Model versions'
      
  SageMakerPipeline:
    Type: AWS::SageMaker::Pipeline
    Properties:
      PipelineName: !Sub '${ModelName}-pipeline'
      PipelineDefinition:
        PipelineDefinitionBody: !Sub |
          {
            "Version": "2020-12-01",
            "Steps": [...]
          }
      RoleArn: !GetAtt SageMakerExecutionRole.Arn

Outputs:
  BucketName:
    Value: !Ref MLBucket
  ModelPackageGroupArn:
    Value: !GetAtt ModelPackageGroup.ModelPackageGroupArn
```

**Terraform Example:**
```hcl
# SageMaker Domain
resource "aws_sagemaker_domain" "ml_domain" {
  domain_name = "ml-platform"
  auth_mode   = "IAM"
  vpc_id      = aws_vpc.main.id
  subnet_ids  = aws_subnet.private[*].id

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn
  }
}

# Model Package Group
resource "aws_sagemaker_model_package_group" "models" {
  model_package_group_name        = "production-models"
  model_package_group_description = "Production ML models"
}

# Feature Store
resource "aws_sagemaker_feature_group" "features" {
  feature_group_name = "customer-features"
  record_identifier_feature_name = "customer_id"
  event_time_feature_name        = "event_time"
  role_arn = aws_iam_role.sagemaker_execution.arn

  online_store_config {
    enable_online_store = true
  }

  offline_store_config {
    s3_storage_config {
      s3_uri = "s3://${aws_s3_bucket.feature_store.id}"
    }
  }

  feature_definition {
    feature_name = "customer_id"
    feature_type = "String"
  }
  
  feature_definition {
    feature_name = "age"
    feature_type = "Integral"
  }
}
```

#### 10.6 Automated Retraining

**When to Retrain:**
- Model performance degradation
- Data drift detected
- Concept drift detected
- Scheduled intervals (weekly, monthly)
- New data available
- Business requirements change

**Automated Retraining Pipeline:**

```python
# EventBridge rule for scheduled retraining
import boto3

events = boto3.client('events')

events.put_rule(
    Name='weekly-model-retrain',
    ScheduleExpression='rate(7 days)',
    State='ENABLED'
)

events.put_targets(
    Rule='weekly-model-retrain',
    Targets=[
        {
            'Arn': 'arn:aws:sagemaker:region:account:pipeline/ml-pipeline',
            'RoleArn': role_arn,
            'Id': '1',
            'SageMakerPipelineParameters': {
                'PipelineParameterList': [
                    {
                        'Name': 'TrainingData',
                        'Value': 's3://bucket/latest-data'
                    }
                ]
            }
        }
    ]
)
```

**Drift-Triggered Retraining:**
```python
# Lambda function triggered by Model Monitor
def lambda_handler(event, context):
    drift_detected = event['detail']['drift_detected']
    
    if drift_detected:
        # Start retraining pipeline
        sagemaker = boto3.client('sagemaker')
        response = sagemaker.start_pipeline_execution(
            PipelineName='ml-pipeline',
            PipelineParameters=[
                {
                    'Name': 'RetryingReason',
                    'Value': 'DataDrift'
                }
            ]
        )
    
    return {'statusCode': 200}
```

#### 10.7 Experiment Tracking and Management

**SageMaker Experiments:**

**Create Experiment:**
```python
from sagemaker.experiments import Experiment, Trial, TrialComponent

# Create experiment
experiment = Experiment.create(
    experiment_name='fraud-detection-experiments',
    description='Testing different algorithms for fraud detection'
)

# Create trial
trial = Trial.create(
    trial_name=f'xgboost-trial-{timestamp}',
    experiment_name=experiment.experiment_name
)

# Run training with tracking
with trial.track() as tracker:
    # Log parameters
    tracker.log_parameters({
        'algorithm': 'xgboost',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100
    })
    
    # Training code here
    model.fit(X_train, y_train)
    
    # Log metrics
    tracker.log_metric('train_accuracy', 0.95)
    tracker.log_metric('val_accuracy', 0.92)
    tracker.log_metric('f1_score', 0.93)
```

**Compare Experiments:**
```python
from sagemaker.analytics import ExperimentAnalytics

analytics = ExperimentAnalytics(
    experiment_name='fraud-detection-experiments'
)

# Get all trials as DataFrame
df = analytics.dataframe()

# Compare metrics
best_trial = df.sort_values('val_accuracy', ascending=False).iloc[0]
print(f"Best model: {best_trial['TrialName']}")
print(f"Accuracy: {best_trial['val_accuracy']}")
```

#### 10.8 Feature Store Operations

**Real-Time Feature Ingestion:**
```python
from sagemaker.feature_store.feature_group import FeatureGroup

feature_group = FeatureGroup(name='customer-features', sagemaker_session=session)

# Ingest record
record = {
    'customer_id': '12345',
    'age': 35,
    'account_balance': 50000.0,
    'transaction_count': 42,
    'event_time': '2024-01-15T10:30:00Z'
}

feature_group.put_record(record=record)
```

**Batch Ingestion:**
```python
# Ingest from DataFrame
import pandas as pd

df = pd.read_csv('customer_features.csv')
feature_group.ingest(data_frame=df, max_workers=3, wait=True)
```

**Online Retrieval:**
```python
# Get features for inference
record = feature_group.get_record(
    record_identifier_value_as_string='12345'
)
```

**Offline Retrieval:**
```python
# Athena query for training
query = f"""
SELECT *
FROM "{feature_group.offline_store_table_name}"
WHERE event_time >= '2024-01-01'
"""

df = feature_group.athena_query(query_string=query).as_dataframe()
```

**Point-in-Time Queries:**
```python
# Get features as they existed at a specific time
from sagemaker.feature_store.feature_store import FeatureStore

fs = FeatureStore(sagemaker_session=session)

features = fs.get_record(
    record_identifier_value_as_string='12345',
    feature_group_name='customer-features',
    as_of_timestamp='2024-01-15T10:00:00Z'
)
```

#### 10.9 Model Governance

**Model Approval Workflow:**

**Automated Approval (based on metrics):**
```python
def approve_model_if_qualified(model_package_arn, min_accuracy=0.90):
    sagemaker = boto3.client('sagemaker')
    
    # Get model metrics
    response = sagemaker.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    metrics = response.get('ModelMetrics', {})
    accuracy = metrics.get('accuracy', 0)
    
    # Auto-approve if meets threshold
    if accuracy >= min_accuracy:
        sagemaker.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        return True
    else:
        # Notify for manual review
        sns.publish(
            TopicArn='arn:aws:sns:...',
            Message=f'Model {model_package_arn} requires manual approval. Accuracy: {accuracy}'
        )
        return False
```

**Manual Approval with SageMaker Studio:**
- Review model metrics
- Compare with previous versions
- Check bias reports (SageMaker Clarify)
- Approve or reject
- Add approval notes

**Multi-Stage Deployment:**
```python
# Stage 1: Dev environment
dev_endpoint = deploy_model(
    model_package_arn,
    endpoint_name='model-dev',
    instance_type='ml.t2.medium'
)

# Stage 2: Staging environment (after dev tests pass)
staging_endpoint = deploy_model(
    model_package_arn,
    endpoint_name='model-staging',
    instance_type='ml.m5.xlarge'
)

# Stage 3: Production (after staging approval)
production_endpoint = deploy_model(
    model_package_arn,
    endpoint_name='model-production',
    instance_type='ml.m5.2xlarge',
    auto_scaling=True
)
```

#### 10.10 MLOps Best Practices

**1. Version Everything:**
- Code (Git)
- Data (S3 versioning, DVC)
- Models (Model Registry)
- Environment (Docker, conda)
- Pipelines (SageMaker Pipelines)

**2. Automate Everything:**
- Testing
- Training
- Evaluation
- Deployment
- Monitoring

**3. Make it Reproducible:**
- Pin dependency versions
- Use seed values
- Document data sources
- Track lineage

**4. Monitor Continuously:**
- Model performance
- Data quality
- System health
- Costs

**5. Test Thoroughly:**
- Unit tests for code
- Integration tests for pipelines
- Model performance tests
- Data validation tests

**Example Test Suite:**
```python
import pytest

def test_data_schema():
    """Validate data schema"""
    df = load_data()
    assert set(df.columns) == set(EXPECTED_COLUMNS)
    assert df['age'].dtype == 'int64'
    assert df.isnull().sum().sum() == 0

def test_model_performance():
    """Ensure model meets minimum performance"""
    model = load_model()
    X_test, y_test = load_test_data()
    
    accuracy = model.score(X_test, y_test)
    assert accuracy >= 0.85, f"Model accuracy {accuracy} below threshold"

def test_prediction_latency():
    """Ensure predictions are fast enough"""
    model = load_model()
    X_sample = load_sample_data()
    
    start = time.time()
    predictions = model.predict(X_sample)
    latency = time.time() - start
    
    assert latency < 0.1, f"Prediction latency {latency}s exceeds 100ms"

def test_model_explainability():
    """Ensure model can be explained"""
    model = load_model()
    explainer = shap.TreeExplainer(model)
    
    X_sample = load_sample_data()
    shap_values = explainer.shap_values(X_sample)
    
    assert shap_values is not None
    assert shap_values.shape[0] == X_sample.shape[0]
```

### Module Summary
- MLOps automates and scales ML workflows
- SageMaker Pipelines orchestrate end-to-end ML workflows
- Model Registry manages versions and approvals
- CI/CD for ML automates testing and deployment
- Infrastructure as Code (CloudFormation, Terraform) ensures reproducibility
- Automated retraining keeps models current
- Feature Store provides consistent feature engineering
- Model governance ensures quality and compliance
- Experiment tracking enables comparison and optimization
- Best practices: version everything, automate, make reproducible, monitor, test

---

## Module 11: Model Performance and Data Quality Monitoring

### Learning Objectives
- Monitor ML models in production
- Detect model and data drift
- Implement automated alerts
- Maintain model quality over time

### Key Topics

#### 11.1 Monitoring Fundamentals

**Why Monitor ML Models?**
- Performance degradation over time
- Data distribution changes
- Concept drift
- System failures
- Compliance requirements
- Cost optimization

**Types of Monitoring:**
1. **Model Performance Monitoring**: Accuracy, latency, throughput
2. **Data Quality Monitoring**: Schema, statistics, drift
3. **Model Quality Monitoring**: Prediction distribution, bias
4. **System Monitoring**: Infrastructure, costs, errors

**Monitoring Architecture:**
```
Production Data → Capture → Analysis → Alerts → Actions
                     ↓          ↓         ↓        ↓
                 S3 Storage  Baseline  CloudWatch Lambda
                            Comparison  SNS      Retrain
```

#### 11.2 SageMaker Model Monitor

**Components:**
1. **Data Capture**: Capture inference inputs/outputs
2. **Baseline**: Establish normal behavior
3. **Monitoring Jobs**: Scheduled comparisons
4. **Violations**: Detected anomalies
5. **Alerts**: Notifications for issues

**Enable Data Capture:**
```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,  # Capture 100% of requests
    destination_s3_uri=f's3://{bucket}/data-capture',
    capture_options=['Input', 'Output']
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='monitored-endpoint',
    data_capture_config=data_capture_config
)
```

#### 11.3 Data Quality Monitoring

**Purpose:**
- Detect schema changes
- Identify missing values
- Find outliers
- Monitor feature distributions

**Create Baseline:**
```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Generate baseline statistics
baseline_job = monitor.suggest_baseline(
    baseline_dataset=f's3://{bucket}/training-data/train.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f's3://{bucket}/baselining',
    wait=True
)

# View baseline statistics
baseline_job.describe()
```

**Schedule Monitoring:**
```python
from sagemaker.model_monitor import CronExpressionGenerator

monitor.create_monitoring_schedule(
    monitor_schedule_name='data-quality-monitor',
    endpoint_input=predictor.endpoint_name,
    output_s3_uri=f's3://{bucket}/monitoring-results',
    statistics=baseline_job.baseline_statistics(),
    constraints=baseline_job.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True
)
```

**Violations Detected:**
- **Data type mismatch**: Expected int, got string
- **Missing features**: Column not present
- **Out of range values**: Value exceeds baseline range
- **Distribution shift**: Feature distribution changed significantly

#### 11.4 Model Quality Monitoring

**Purpose:**
- Monitor prediction quality
- Detect accuracy degradation
- Track business metrics

**Setup (requires ground truth labels):**
```python
from sagemaker.model_monitor import ModelQualityMonitor

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Create baseline
baseline_job = model_quality_monitor.suggest_baseline(
    baseline_dataset=f's3://{bucket}/validation-data-with-labels.csv',
    dataset_format=DatasetFormat.csv(header=True),
    problem_type='BinaryClassification',  # or 'MulticlassClassification', 'Regression'
    inference_attribute='prediction',
    ground_truth_attribute='label',
    output_s3_uri=f's3://{bucket}/model-quality-baseline'
)

# Schedule monitoring
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name='model-quality-monitor',
    endpoint_input=predictor.endpoint_name,
    ground_truth_input=f's3://{bucket}/ground-truth/',  # Labels from your system
    problem_type='BinaryClassification',
    constraints=baseline_job.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.daily(),
    enable_cloudwatch_metrics=True
)
```

**Metrics Monitored:**
- **Classification**: Accuracy, Precision, Recall, F1, AUC
- **Regression**: MAE, MSE, RMSE, R²
- **Ranking**: NDCG, MRR

#### 11.5 Model Drift Detection

**Types of Drift:**

**1. Data Drift (Covariate Shift):**
- Input distribution changes
- Features have different statistics
- Example: Age distribution shifts from [20-40] to [40-60]

**2. Concept Drift:**
- Relationship between inputs and outputs changes
- Example: Customer behavior changes during pandemic

**3. Label Drift:**
- Target variable distribution changes
- Example: Fraud rate increases from 1% to 5%

**Detecting Data Drift:**

**Statistical Tests:**
- **Kolmogorov-Smirnov test**: Compare distributions
- **Chi-square test**: Categorical variables
- **Population Stability Index (PSI)**: Overall drift measure

**PSI Calculation:**
```python
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    """
    # Bin the data
    breakpoints = np.percentile(expected, np.arange(0, 100, 100/bins))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi

# Example usage
baseline_data = load_baseline_features()
current_data = load_current_features()

psi = calculate_psi(baseline_data['age'], current_data['age'])

if psi >= 0.2:
    print("Significant drift detected! Consider retraining.")
```

**SageMaker Clarify for Drift:**
```python
from sagemaker.clarify import DataConfig, BiasConfig, ModelPredictedLabelConfig
from sagemaker import clarify

# Configure drift detection
data_config = DataConfig(
    s3_data_input_path=f's3://{bucket}/current-data',
    s3_output_path=f's3://{bucket}/drift-analysis',
    label='target',
    headers=feature_names,
    dataset_type='text/csv'
)

drift_config = clarify.DriftCheckConfig(
    baseline_s3_uri=f's3://{bucket}/baseline-data',
    s3_analysis_config_output_path=f's3://{bucket}/drift-config'
)

clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

clarify_processor.run_pre_training_bias(
    data_config=data_config,
    data_bias_config=bias_config
)
```

#### 11.6 Bias and Fairness Monitoring

**SageMaker Clarify for Bias Detection:**

**Pre-training Bias:**
```python
from sagemaker.clarify import BiasConfig

bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name='gender',  # Sensitive attribute
    facet_values_or_threshold=[0]  # Reference group
)

clarify_processor.run_pre_training_bias(
    data_config=data_config,
    data_bias_config=bias_config,
    methods='all'
)
```

**Post-training Bias:**
```python
model_config = ModelConfig(
    model_name=model.name,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    accept_type='text/csv'
)

predictions_config = ModelPredictedLabelConfig(
    probability_threshold=0.5
)

clarify_processor.run_post_training_bias(
    data_config=data_config,
    data_bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=predictions_config
)
```

**Bias Metrics:**
- **Class Imbalance (CI)**: Distribution across groups
- **Difference in Positive Proportions in Labels (DPL)**
- **Disparate Impact (DI)**: Ratio of outcomes
- **Equal Opportunity Difference (EOD)**: Recall difference
- **Treatment Equality (TE)**: Error rate difference

**Bias Drift Monitoring:**
```python
from sagemaker.model_monitor import BiasMonitor

bias_monitor = BiasMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

bias_monitor.create_monitoring_schedule(
    monitor_schedule_name='bias-drift-monitor',
    endpoint_input=predictor.endpoint_name,
    ground_truth_input=f's3://{bucket}/ground-truth/',
    constraints=bias_baseline.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.daily(),
    enable_cloudwatch_metrics=True
)
```

#### 11.7 Feature Attribution Drift

**Purpose:**
- Monitor changes in feature importance
- Detect if model relies on different features
- Identify potential issues

**SHAP Baseline:**
```python
from sagemaker.clarify import SHAPConfig

shap_config = SHAPConfig(
    baseline=[X_train.median().values.tolist()],
    num_samples=100,
    agg_method='mean_abs'
)

explainability_config = clarify.ExplainabilityConfig(
    shap_config=shap_config
)

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=explainability_config
)
```

**Monitor Feature Attribution:**
```python
from sagemaker.model_monitor import FeatureAttributionMonitor

feature_attribution_monitor = FeatureAttributionMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

feature_attribution_monitor.create_monitoring_schedule(
    monitor_schedule_name='feature-attribution-monitor',
    endpoint_input=predictor.endpoint_name,
    explainability_config=explainability_baseline.config,
    schedule_cron_expression=CronExpressionGenerator.daily()
)
```

#### 11.8 Alerting and Notifications

**CloudWatch Alarms:**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Alarm for data quality violations
cloudwatch.put_metric_alarm(
    AlarmName='DataQualityViolations',
    MetricName='feature_baseline_drift_age',
    Namespace='aws/sagemaker/Endpoints/data-metrics',
    Statistic='Average',
    Period=3600,
    EvaluationPeriods=1,
    Threshold=0.2,  # PSI threshold
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=[
        'arn:aws:sns:region:account:ml-alerts'
    ],
    AlarmDescription='Alert when age feature drifts significantly'
)

# Alarm for model accuracy drop
cloudwatch.put_metric_alarm(
    AlarmName='ModelAccuracyDrop',
    MetricName='accuracy',
    Namespace='aws/sagemaker/Endpoints/model-metrics',
    Statistic='Average',
    Period=3600,
    EvaluationPeriods=2,
    Threshold=0.85,
    ComparisonOperator='LessThanThreshold',
    AlarmActions=[
        'arn:aws:sns:region:account:ml-alerts'
    ],
    AlarmDescription='Alert when model accuracy drops below 85%'
)
```

**SNS Notifications:**
```python
sns = boto3.client('sns')

# Create topic
topic = sns.create_topic(Name='ml-monitoring-alerts')

# Subscribe email
sns.subscribe(
    TopicArn=topic['TopicArn'],
    Protocol='email',
    Endpoint='ml-team@company.com'
)

# Subscribe Lambda for automated actions
sns.subscribe(
    TopicArn=topic['TopicArn'],
    Protocol='lambda',
    Endpoint='arn:aws:lambda:region:account:function:handle-drift'
)
```

**Automated Response Lambda:**
```python
def lambda_handler(event, context):
    """
    Triggered when drift detected
    Actions:
    1. Log to CloudWatch
    2. Start retraining pipeline
    3. Notify team
    """
    message = json.loads(event['Records'][0]['Sns']['Message'])
    
    if 'DataDrift' in message['AlarmName']:
        # Start retraining
        sagemaker = boto3.client('sagemaker')
        response = sagemaker.start_pipeline_execution(
            PipelineName='ml-retraining-pipeline',
            PipelineParameters=[
                {
                    'Name': 'Reason',
                    'Value': 'DataDrift'
                }
            ]
        )
        
        # Log
        print(f"Started retraining: {response['PipelineExecutionArn']}")
        
        # Notify team
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:region:account:ml-team',
            Subject='Model Retraining Started',
            Message=f"Data drift detected. Retraining started: {response['PipelineExecutionArn']}"
        )
    
    return {'statusCode': 200}
```

#### 11.9 Model Performance Dashboards

**CloudWatch Dashboard:**
```python
import boto3
import json

cloudwatch = boto3.client('cloudwatch')

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", {"stat": "Average"}],
                    [".", "Invocations", {"stat": "Sum"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Endpoint Performance"
            }
        },
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["aws/sagemaker/Endpoints/data-metrics", "feature_baseline_drift_age"],
                    [".", "feature_baseline_drift_income"]
                ],
                "period": 3600,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Feature Drift"
            }
        },
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["aws/sagemaker/Endpoints/model-metrics", "accuracy"],
                    [".", "precision"],
                    [".", "recall"]
                ],
                "period": 3600,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Model Quality"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='ML-Model-Monitoring',
    DashboardBody=json.dumps(dashboard_body)
)
```

**Custom Metrics:**
```python
def publish_custom_metrics(predictions, labels):
    """Publish custom business metrics to CloudWatch"""
    cloudwatch = boto3.client('cloudwatch')
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    # Publish
    cloudwatch.put_metric_data(
        Namespace='Custom/ML/Model',
        MetricData=[
            {
                'MetricName': 'Accuracy',
                'Value': accuracy,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'Precision',
                'Value': precision,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'Recall',
                'Value': recall,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            }
        ]
    )
```

#### 11.10 Monitoring Best Practices

**1. Establish Baselines Early:**
- Create baselines during training
- Use representative data
- Update baselines periodically

**2. Monitor Multiple Dimensions:**
- Data quality
- Model performance
- System health
- Business metrics
- Bias and fairness

**3. Set Appropriate Thresholds:**
- Start conservative
- Adjust based on false positive/negative rates
- Consider business impact

**4. Automate Responses:**
- Alert for critical issues
- Auto-retrain for drift
- Rollback for severe degradation
- Log all actions

**5. Regular Reviews:**
- Weekly metric reviews
- Monthly model performance analysis
- Quarterly baseline updates
- Annual monitoring strategy review

**6. Cost Monitoring:**
```python
# Monitor endpoint costs
def calculate_endpoint_cost(endpoint_name, hours=24):
    cloudwatch = boto3.client('cloudwatch')
    
    # Get instance hours
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='InstanceCount',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name}
        ],
        StartTime=datetime.utcnow() - timedelta(hours=hours),
        EndTime=datetime.utcnow(),
        Period=3600,
        Statistics=['Average']
    )
    
    # Calculate cost (example: ml.m5.xlarge = $0.23/hr)
    instance_price = 0.23
    total_hours = sum([point['Average'] for point in response['Datapoints']])
    total_cost = total_hours * instance_price
    
    return total_cost
```

### Module Summary
- Monitor model performance, data quality, and system health
- Use SageMaker Model Monitor for automated monitoring
- Detect data drift, concept drift, and label drift
- Monitor bias and fairness with SageMaker Clarify
- Track feature attribution changes
- Set up CloudWatch alarms and SNS notifications
- Automate responses to drift and degradation
- Create dashboards for visibility
- Establish baselines and update regularly
- Monitor costs alongside performance

---

## Module 12: Course Summary

### Course Recap

**Module 0: Introduction**
- AWS ML stack overview
- SageMaker ecosystem
- ML lifecycle

**Module 1: ML on AWS**
- ML fundamentals
- AWS ML services
- SageMaker components

**Module 2: Business Challenges**
- Problem framing
- Success metrics
- Data requirements
- Feasibility assessment

**Module 3: Data Processing**
- Data ingestion patterns
- AWS data services (S3, Glue, Athena, EMR, Kinesis)
- Data lake architecture
- Data quality

**Module 4: Feature Engineering**
- Numerical transformations
- Categorical encoding
- Text processing
- Feature Store
- Feature selection

**Module 5: Model Selection**
- Algorithm types
- Selection criteria
- SageMaker built-in algorithms
- Interpretability vs complexity

**Module 6: Model Training**
- Training workflows
- Hyperparameter tuning
- Distributed training
- Cost optimization
- Debugging and profiling

**Module 7: Evaluation and Tuning**
- Classification/regression metrics
- Cross-validation
- Overfitting/underfitting
- Regularization
- Imbalanced data
- Model explainability

**Module 8: Model Deployment**
- Inference options (real-time, batch, serverless, async)
- Model optimization (Neo, quantization)
- Auto-scaling
- Inference pipelines
- Edge deployment

**Module 9: Security**
- IAM and access control
- Encryption (at rest, in transit)
- VPC configuration
- Logging and monitoring
- Compliance and governance

**Module 10: MLOps**
- SageMaker Pipelines
- Model Registry
- CI/CD for ML
- Infrastructure as Code
- Automated retraining
- Feature Store operations

**Module 11: Monitoring**
- Data quality monitoring
- Model quality monitoring
- Drift detection (data, concept, label)
- Bias monitoring
- Alerting and automation

### Key AWS Services Summary

| Service | Purpose | Use When |
|---------|---------|----------|
| SageMaker Studio | ML development environment | Building and experimenting |
| SageMaker Pipelines | ML workflow orchestration | Automating ML workflows |
| SageMaker Training | Model training | Training ML models |
| SageMaker Inference | Model deployment | Deploying models |
| SageMaker Feature Store | Feature management | Sharing features across teams |
| SageMaker Model Monitor | Production monitoring | Detecting drift and issues |
| SageMaker Clarify | Bias detection, explainability | Ensuring fairness |
| SageMaker Autopilot | AutoML | Quick model development |
| S3 | Data storage | Storing datasets, models |
| AWS Glue | ETL | Processing and cataloging data |
| Athena | SQL queries | Analyzing data in S3 |
| EMR | Big data processing | Large-scale data processing |
| Kinesis | Streaming data | Real-time data ingestion |
| CloudWatch | Monitoring and logging | Observability |
| CloudTrail | API logging | Auditing |
| IAM | Access control | Security |
| KMS | Encryption key management | Data protection |

### Exam Preparation Tips

**MLA-C01 Exam Domains:**

**Domain 1: Data Engineering (20%)**
- Data ingestion strategies
- Data transformation
- Feature engineering
- Data quality

**Domain 2: Exploratory Data Analysis (24%)**
- Data visualization
- Statistical analysis
- Feature selection
- Data preparation

**Domain 3: Modeling (36%)**
- Algorithm selection
- Training and tuning
- Evaluation
- Model optimization

**Domain 4: ML Implementation and Operations (20%)**
- Deployment
- Monitoring
- Security
- MLOps

**Study Recommendations:**
1. Hands-on practice with SageMaker
2. Understand when to use each service
3. Know algorithm selection criteria
4. Practice with sample datasets
5. Review AWS documentation
6. Take practice exams
7. Build end-to-end ML projects

**Common Exam Topics:**
- SageMaker built-in algorithms
- Hyperparameter tuning strategies
- Deployment options
- Feature engineering techniques
- Model evaluation metrics
- Cost optimization
- Security best practices
- Monitoring and drift detection

### Next Steps

**Hands-On Practice:**
1. Complete SageMaker Studio Labs
2. Build an end-to-end ML project
3. Deploy a model to production
4. Set up monitoring and alerts
5. Implement ML pipeline

**Additional Resources:**
- AWS ML Training and Certification
- SageMaker Examples GitHub repository
- AWS Whitepapers on ML
- AWS ML Blog
- AWS ML Community

**Certification Path:**
- AWS Certified Machine Learning - Specialty (MLA-C01)
- Continue with AWS Certified Solutions Architect
- Or AWS Certified Data Analytics

### Conclusion

This course covered the complete ML engineering lifecycle on AWS:
- Data processing and feature engineering
- Model training, evaluation, and optimization
- Deployment and inference
- Security and compliance
- MLOps and automation
- Monitoring and maintenance

**Key Takeaways:**
- AWS provides comprehensive ML services
- Automation is crucial for production ML
- Security and monitoring are essential
- MLOps practices improve reliability
- Continuous learning and improvement

**Good luck with your AWS Machine Learning Engineer certification!**

---

## Appendix: Quick Reference

### Common SageMaker Code Patterns

**Training:**
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='training-image',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1
)

estimator.fit({'training': 's3://bucket/train'})
```

**Deployment:**
```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

result = predictor.predict(data)
```

**Pipeline:**
```python
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name='ml-pipeline',
    steps=[processing_step, training_step, evaluation_step]
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

**Monitoring:**
```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(role=role, instance_type='ml.m5.xlarge')

monitor.create_monitoring_schedule(
    monitor_schedule_name='monitor',
    endpoint_input=predictor.endpoint_name,
    schedule_cron_expression='cron(0 * * * ? *)'
)
```

### Important Formulas

**Evaluation Metrics:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- RMSE = √[(1/n) * Σ(actual - predicted)²]

**Regularization:**
- L1: Loss + λ * Σ|weights|
- L2: Loss + λ * Σ(weights²)

**Learning Rate:**
- Too high: Unstable, oscillation
- Too low: Slow convergence
- Typical: 0.001 to 0.1

---

**End of Study Guide**
