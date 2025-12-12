# AWS Machine Learning Engineering on AWS - Study Guide (Part 2)

> **Continuation of Modules 7-12**

---

## Module 7: Model Evaluation and Tuning

### Learning Objectives
- Understand model evaluation metrics
- Learn validation strategies
- Diagnose and fix model problems
- Optimize model performance

### Key Topics

#### 7.1 Model Evaluation Fundamentals

**Why Evaluate?**
- Measure model quality
- Compare different models
- Identify problems (bias, variance)
- Ensure generalization to new data
- Make informed deployment decisions

**Evaluation Process:**
1. Split data (train/validation/test)
2. Train model on training data
3. Tune on validation data
4. Final evaluation on test data
5. Never touch test data until final evaluation

#### 7.2 Classification Metrics

**Confusion Matrix:**

```
                 Predicted
              Positive  Negative
Actual  Pos     TP        FN
        Neg     FP        TN
```

**Basic Metrics:**

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Overall correctness
- Use when: Balanced classes
- Don't use when: Imbalanced data

**Precision:**
```
Precision = TP / (TP + FP)
```
- Of positive predictions, how many are correct?
- Use when: False positives are costly
- Example: Spam detection (don't want to mark good emails as spam)

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
- Of actual positives, how many did we find?
- Use when: False negatives are costly
- Example: Disease detection (don't want to miss sick patients)

**F1 Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balances both metrics
- Use when: Need balance between precision and recall

**Specificity:**
```
Specificity = TN / (TN + FP)
```
- True negative rate
- Use when: Correctly identifying negatives matters

**Advanced Metrics:**

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
- Plots True Positive Rate vs False Positive Rate
- AUC = 0.5: Random classifier
- AUC = 1.0: Perfect classifier
- Use when: Comparing models, threshold-independent metric
- Good for imbalanced data

**Precision-Recall Curve:**
- Plots Precision vs Recall
- Better than ROC for imbalanced data
- Use when: Positive class is rare

**Multi-Class Metrics:**
- **Macro Average**: Average across classes (treats all classes equally)
- **Micro Average**: Aggregate across all classes (favors frequent classes)
- **Weighted Average**: Weighted by class support

#### 7.3 Regression Metrics

**Mean Absolute Error (MAE):**
```
MAE = (1/n) * Σ|actual - predicted|
```
- Average absolute difference
- Same units as target variable
- Robust to outliers
- Easy to interpret

**Mean Squared Error (MSE):**
```
MSE = (1/n) * Σ(actual - predicted)²
```
- Penalizes large errors more
- Not robust to outliers
- Common loss function

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE
```
- Same units as target
- Penalizes large errors
- Most popular regression metric

**R² (R-squared, Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
```
- Proportion of variance explained
- Range: 0 to 1 (can be negative for poor models)
- 1.0 = perfect predictions
- Use when: Want to understand variance explained

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/n) * Σ|actual - predicted| / |actual|
```
- Percentage error
- Scale-independent
- Use when: Comparing different scales
- Problem: Undefined when actual = 0

**Choosing Regression Metrics:**
- **MAE**: Robust, easy to explain
- **RMSE**: Penalize large errors
- **R²**: Variance explained
- **MAPE**: Scale-independent comparison

#### 7.4 Validation Strategies

**Train-Test Split:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- Simple and fast
- Use when: Large dataset (>10K samples)
- Risk: High variance if small dataset

**K-Fold Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```
- Split data into K folds
- Train on K-1 folds, validate on 1
- Repeat K times
- Average results
- Use when: Small-medium datasets
- More reliable than single split

**Stratified K-Fold:**
- Maintains class distribution in each fold
- Use when: Imbalanced classes
- Important for classification

**Time Series Validation:**
- Never shuffle time-series data
- Use TimeSeriesSplit
- Train on past, predict future
- Rolling window approach

**Leave-One-Out (LOO):**
- K = number of samples
- Very computationally expensive
- Use when: Very small datasets (<100 samples)

#### 7.5 Overfitting and Underfitting

**Bias-Variance Tradeoff:**

```
Total Error = Bias² + Variance + Irreducible Error
```

**Underfitting (High Bias):**
- Model too simple
- Poor performance on both train and test
- Symptoms:
  - Low training accuracy
  - Low test accuracy
  - Large error
- Solutions:
  - Increase model complexity
  - Add more features
  - Reduce regularization
  - Train longer

**Overfitting (High Variance):**
- Model too complex
- Memorizes training data
- Symptoms:
  - High training accuracy
  - Low test accuracy
  - Large gap between train and test
- Solutions:
  - Get more data
  - Reduce model complexity
  - Increase regularization
  - Use dropout
  - Early stopping
  - Data augmentation
  - Cross-validation

**Good Fit:**
- Training accuracy: ~90%
- Test accuracy: ~88%
- Small gap between train and test

**Detecting Overfitting:**
```python
# Plot learning curves
train_scores = []
val_scores = []

for size in training_sizes:
    # Train model
    train_score = evaluate(model, train_data)
    val_score = evaluate(model, val_data)
    train_scores.append(train_score)
    val_scores.append(val_score)

# Overfitting if: train_score >> val_score
```

#### 7.6 Regularization Techniques

**L1 Regularization (Lasso):**
```
Loss = Original Loss + λ * Σ|weights|
```
- Adds absolute value of weights
- Feature selection (drives weights to zero)
- Sparse models
- Use when: Many irrelevant features

**L2 Regularization (Ridge):**
```
Loss = Original Loss + λ * Σ(weights²)
```
- Adds square of weights
- Shrinks weights toward zero
- Keeps all features
- Use when: Multicollinearity, overfitting

**Elastic Net:**
```
Loss = Original Loss + λ₁*Σ|weights| + λ₂*Σ(weights²)
```
- Combination of L1 and L2
- Best of both worlds
- Use when: Many features, some correlated

**Dropout (Neural Networks):**
- Randomly drop neurons during training
- Prevents co-adaptation
- Typical values: 0.2 to 0.5
- Use when: Deep neural networks

**Early Stopping:**
- Stop training when validation loss stops improving
- Monitor validation metric
- Save best model
- Patience parameter (wait N epochs)

**Data Augmentation:**
- Create synthetic training samples
- Images: rotation, flip, crop, color jitter
- Text: synonym replacement, back-translation
- Increases dataset size
- Improves generalization

#### 7.7 Model Tuning Strategies

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
```
- Exhaustive search
- Tries all combinations
- Expensive but thorough
- Use when: Few hyperparameters

**Random Search:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.001, 0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200, 500]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5
)
```
- Random sampling
- More efficient than grid search
- Use when: Many hyperparameters, limited time

**Bayesian Optimization:**
- Intelligent search
- Uses previous results
- Probabilistic model of objective function
- SageMaker Automatic Model Tuning uses this
- Most efficient for expensive evaluations

**Tuning Process:**
1. Start with default parameters
2. Identify important hyperparameters
3. Define search space (wide initially)
4. Run tuning (random or Bayesian)
5. Narrow search space
6. Fine-tune around best values

#### 7.8 Handling Imbalanced Data

**Problem:**
- Rare class underrepresented
- Model biased toward majority class
- High accuracy but poor recall on minority class

**Detection:**
- Check class distribution
- Accuracy is high but precision/recall imbalanced
- Confusion matrix shows bias

**Solutions:**

**1. Resampling:**

**Undersampling:**
- Remove majority class samples
- Fast training
- Risk: Loss of information
- Use when: Large dataset

**Oversampling:**
- Duplicate minority class samples
- Risk: Overfitting
- Use when: Small minority class

**SMOTE (Synthetic Minority Oversampling):**
- Creates synthetic samples
- Interpolates between minority samples
- Better than simple duplication
- Recommended approach

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**2. Algorithm-Level:**

**Class Weights:**
- Penalize misclassifying minority class more
- Available in most algorithms

```python
model = RandomForestClassifier(
    class_weight='balanced'  # or custom weights
)
```

**Threshold Adjustment:**
- Default threshold: 0.5
- Adjust based on precision-recall curve
- Lower threshold increases recall
- Higher threshold increases precision

**3. Evaluation:**
- Don't use accuracy
- Use: Precision, Recall, F1, AUC-ROC
- Focus on minority class performance
- Confusion matrix analysis

**4. Ensemble Methods:**
- BalancedBaggingClassifier
- BalancedRandomForestClassifier
- EasyEnsemble
- Combines resampling with ensemble

#### 7.9 Model Interpretability and Explainability

**Why Explainability Matters:**
- Build trust
- Debug models
- Regulatory compliance
- Ethical AI
- Identify biases

**Techniques:**

**Feature Importance:**
```python
# Tree-based models
importance = model.feature_importances_

# Permutation importance (any model)
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val)
```

**SHAP (SHapley Additive exPlanations):**
- Game theory approach
- Feature contribution to each prediction
- Local and global explanations
- Model-agnostic

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

**LIME (Local Interpretable Model-agnostic Explanations):**
- Explains individual predictions
- Approximates locally with simple model
- Works with any model

**Partial Dependence Plots:**
- Shows effect of feature on prediction
- Marginalizes over other features
- Visualizes relationships

**SageMaker Clarify:**
- Bias detection
- Feature importance
- SHAP values
- Explainability reports
- Integrates with training

```python
from sagemaker import clarify

explainability_config = clarify.SHAPConfig(
    baseline=[X_train.mean().values.tolist()],
    num_samples=100,
    agg_method='mean_abs'
)

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=explainability_config
)
```

#### 7.10 A/B Testing Models

**Purpose:**
- Compare models in production
- Gradual rollout
- Measure real-world performance
- Reduce risk

**Setup:**
- Control group: Existing model
- Treatment group: New model
- Random assignment
- Monitor metrics

**SageMaker Implementation:**

**Multi-Variant Endpoints:**
```python
from sagemaker.model import Model

# Deploy multiple variants
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='ab-test-endpoint',
    variant_name='ModelA',
    traffic_distribution={'ModelA': 0.5, 'ModelB': 0.5}
)
```

**Gradual Rollout:**
- Start: 95% control, 5% treatment
- Monitor performance
- Gradually increase treatment traffic
- Rollback if issues detected

**Metrics to Monitor:**
- Prediction accuracy
- Latency
- Error rate
- Business metrics (revenue, conversions)

### Module Summary
- Evaluate models with appropriate metrics (accuracy, precision, recall, F1, RMSE, R²)
- Use cross-validation for reliable estimates
- Diagnose overfitting (high train, low test) and underfitting (low train, low test)
- Apply regularization (L1, L2, dropout, early stopping)
- Tune hyperparameters with grid search, random search, or Bayesian optimization
- Handle imbalanced data with resampling, class weights, or threshold adjustment
- Explain models with SHAP, LIME, feature importance
- A/B test models in production for safe deployment

---

## Module 8: Model Deployment

### Learning Objectives
- Deploy ML models to production
- Understand inference options on AWS
- Implement real-time and batch inference
- Optimize inference performance and cost

### Key Topics

#### 8.1 Deployment Fundamentals

**ML Deployment Challenges:**
- Low latency requirements
- High availability
- Scalability
- Cost optimization
- Model versioning
- Monitoring and logging
- Security

**Inference Types:**

**Real-Time Inference:**
- Synchronous predictions
- Low latency (milliseconds)
- Single or small batch
- Use cases: Fraud detection, recommendations, chatbots

**Batch Inference:**
- Asynchronous predictions
- Large datasets
- Not time-critical
- Use cases: Daily reports, bulk scoring

**Asynchronous Inference:**
- Queue-based
- Longer processing time
- Large payloads
- Use cases: Document processing, video analysis

#### 8.2 SageMaker Inference Options

**1. Real-Time Endpoints:**

**Deployment:**
```python
from sagemaker.model import Model

model = Model(
    model_data='s3://bucket/model.tar.gz',
    image_uri=image_uri,
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-endpoint'
)
```

**Prediction:**
```python
result = predictor.predict(data)
```

**Features:**
- Persistent endpoint
- Auto-scaling
- Multi-model endpoints
- Multi-variant endpoints (A/B testing)
- Data capture for monitoring

**Instance Selection:**
- **CPU instances (ml.m5, ml.c5)**: Cost-effective, general models
- **GPU instances (ml.p3, ml.g4dn)**: Deep learning, large models
- **Inference-optimized (ml.inf1)**: AWS Inferentia, cost-effective
- **Graviton (ml.m6g)**: ARM-based, price-performance

**2. Serverless Inference:**

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10
)

predictor = model.deploy(
    serverless_inference_config=serverless_config
)
```

**Benefits:**
- No instance management
- Auto-scaling (including to zero)
- Pay per use
- Use when: Intermittent traffic, low volume

**Limitations:**
- Cold starts (seconds)
- Max 4GB memory
- Max 60s processing time
- Max 6MB payload

**3. Batch Transform:**

```python
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/output'
)

transformer.transform(
    data='s3://bucket/input',
    content_type='text/csv',
    split_type='Line'
)
```

**Features:**
- Process large datasets
- No persistent endpoint
- Parallel processing
- Cost-effective for bulk
- Use when: Scheduled jobs, large batches

**4. Asynchronous Inference:**

```python
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path='s3://bucket/output',
    max_concurrent_invocations_per_instance=10
)

predictor = model.deploy(
    async_inference_config=async_config,
    instance_type='ml.m5.xlarge',
    initial_instance_count=1
)
```

**Features:**
- Queue-based
- Large payloads (up to 1GB)
- Long processing time (up to 15 min)
- SNS notifications
- Auto-scaling to zero
- Use when: Variable traffic, large payloads

**Comparison:**

| Option | Latency | Payload | Scaling | Cost | Use Case |
|--------|---------|---------|---------|------|----------|
| Real-Time | ms | <6MB | Manual/Auto | High | Interactive apps |
| Serverless | seconds | <6MB | Auto to zero | Low | Intermittent |
| Batch | minutes-hours | Unlimited | Cluster | Medium | Scheduled jobs |
| Async | seconds-minutes | <1GB | Auto to zero | Medium | Variable traffic |

#### 8.3 Model Optimization for Inference

**Model Compilation:**

**SageMaker Neo:**
- Compiles model for specific hardware
- Up to 2x performance improvement
- Reduces model size
- Supports TensorFlow, PyTorch, MXNet, XGBoost
- Target devices: Cloud, edge, IoT

```python
from sagemaker.neo import NeoModel

neo_model = NeoModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework='tensorflow',
    framework_version='2.9',
    py_version='py3',
    image_uri=image_uri,
    model_input_shape={'data': [1, 224, 224, 3]},
    target_instance_family='ml_m5'
)

neo_model.compile()
```

**Quantization:**
- Reduce precision (FP32 → INT8)
- Smaller model size
- Faster inference
- Minimal accuracy loss
- 2-4x speedup

**Pruning:**
- Remove unnecessary weights
- Smaller model size
- Faster inference
- May require retraining

**Knowledge Distillation:**
- Train small model from large model
- Student-teacher approach
- Maintains accuracy
- Reduces size and latency

#### 8.4 Multi-Model Endpoints

**Purpose:**
- Host multiple models on single endpoint
- Cost optimization
- Dynamic model loading
- Use when: Many similar models

**Example:**
```python
from sagemaker.multidatamodel import MultiDataModel

multi_model = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix='s3://bucket/models/',
    image_uri=image_uri,
    role=role
)

predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Predict with specific model
result = predictor.predict(data, target_model='model1.tar.gz')
```

**Benefits:**
- Up to 70% cost savings
- Shared infrastructure
- Dynamic loading/unloading
- LRU cache management

**Limitations:**
- Models must use same framework
- 5GB max per model
- Some frameworks not supported

#### 8.5 Auto-Scaling

**Why Auto-Scale?**
- Handle variable traffic
- Cost optimization
- Maintain performance
- Automatic adjustment

**Configuration:**
```python
import boto3

client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Define scaling policy
client.put_scaling_policy(
    PolicyName='scale-on-invocations',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1000.0,  # Target invocations per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

**Metrics for Scaling:**
- InvocationsPerInstance
- ModelLatency
- CPUUtilization
- MemoryUtilization

**Best Practices:**
- Set appropriate min/max capacity
- Use scale-out faster than scale-in
- Consider warm-up time
- Test scaling behavior
- Monitor costs

#### 8.6 Inference Pipelines

**Purpose:**
- Combine preprocessing and inference
- Reduce latency
- Simplify deployment
- Ensure consistency

**Example:**
```python
from sagemaker.pipeline import PipelineModel

# Preprocessing container
preprocess_model = Model(
    model_data='s3://bucket/preprocess.tar.gz',
    image_uri=preprocess_image_uri,
    role=role
)

# Inference container
inference_model = Model(
    model_data='s3://bucket/model.tar.gz',
    image_uri=inference_image_uri,
    role=role
)

# Create pipeline
pipeline_model = PipelineModel(
    name='inference-pipeline',
    role=role,
    models=[preprocess_model, inference_model]
)

predictor = pipeline_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

**Benefits:**
- Single API call
- Reduced network latency
- Feature engineering consistency
- Simplified client code

**Use Cases:**
- Feature scaling/normalization
- Text tokenization
- Image preprocessing
- Ensemble models

#### 8.7 Edge Deployment

**AWS IoT Greengrass + SageMaker:**
- Deploy models to edge devices
- Local inference
- Low latency
- Offline operation
- Bandwidth savings

**SageMaker Edge Manager:**
- Model packaging for edge
- Fleet management
- Model updates
- Monitoring and sampling
- Data collection

**SageMaker Neo for Edge:**
- Optimize for edge hardware
- Raspberry Pi, Jetson Nano, etc.
- Minimal dependencies
- Small footprint

**Deployment Process:**
1. Train model in SageMaker
2. Compile with Neo for target device
3. Package for edge
4. Deploy via IoT Greengrass
5. Monitor with Edge Manager

#### 8.8 Model Deployment Best Practices

**Version Control:**
- Track model versions
- Store metadata
- Enable rollback
- Use SageMaker Model Registry

**Blue/Green Deployment:**
- Deploy new version alongside old
- Gradually shift traffic
- Rollback if issues
- Zero downtime

**Canary Deployment:**
- Route small percentage to new model
- Monitor performance
- Gradually increase
- Early issue detection

**Shadow Mode:**
- New model runs in parallel
- Predictions logged, not served
- Compare with production
- Safe testing

**Monitoring:**
- Prediction latency
- Throughput
- Error rates
- Model drift
- Data quality

**Security:**
- VPC deployment
- Encryption in transit (TLS)
- Encryption at rest
- IAM roles and policies
- Network isolation

**Cost Optimization:**
- Right-size instances
- Use auto-scaling
- Use Spot instances (if appropriate)
- Multi-model endpoints
- Serverless for low traffic
- Delete unused endpoints

### Module Summary
- Choose inference type: real-time, serverless, batch, or async
- Optimize models with Neo, quantization, pruning
- Use multi-model endpoints for cost savings
- Implement auto-scaling for variable traffic
- Deploy inference pipelines for preprocessing
- Deploy to edge with IoT Greengrass and Edge Manager
- Follow best practices: versioning, blue/green, monitoring, security

---

## Module 9: Securing AWS ML Resources

### Learning Objectives
- Implement security best practices for ML workloads
- Understand AWS security services for ML
- Configure encryption and access control
- Ensure compliance and governance

### Key Topics

#### 9.1 Security Fundamentals

**Shared Responsibility Model:**
- **AWS Responsibility**: Security OF the cloud
  - Physical security
  - Infrastructure
  - Hardware
  - Network
  - Hypervisor

- **Customer Responsibility**: Security IN the cloud
  - Data encryption
  - Access control
  - Network configuration
  - Application security
  - Compliance

**Security Pillars for ML:**
1. **Identity and Access Management**
2. **Data Protection**
3. **Infrastructure Security**
4. **Logging and Monitoring**
5. **Compliance**

#### 9.2 Identity and Access Management (IAM)

**IAM Principals:**
- Users
- Groups
- Roles (recommended for AWS services)
- Federated users

**Least Privilege Principle:**
- Grant minimum permissions needed
- Use managed policies when possible
- Regular access reviews
- Time-bound access

**SageMaker Execution Roles:**

```python
# Example IAM policy for SageMaker
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-sagemaker-bucket/*",
                "arn:aws:s3:::my-sagemaker-bucket"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

**Service Control Policies (SCPs):**
- Organization-level controls
- Maximum permissions boundary
- Apply to accounts and OUs
- Use for governance

**IAM Conditions:**
- Source IP restrictions
- MFA requirements
- Time-based access
- Resource tagging

**Example with Conditions:**
```json
{
    "Effect": "Allow",
    "Action": "sagemaker:CreateTrainingJob",
    "Resource": "*",
    "Condition": {
        "StringEquals": {
            "aws:RequestedRegion": "us-east-1"
        },
        "IpAddress": {
            "aws:SourceIp": "203.0.113.0/24"
        }
    }
}
```

#### 9.3 Data Protection

**Encryption at Rest:**

**S3 Encryption:**
- **SSE-S3**: S3-managed keys
- **SSE-KMS**: AWS KMS keys (recommended)
- **SSE-C**: Customer-provided keys
- Default encryption enabled

```python
# SageMaker with KMS encryption
from sagemaker.estimator import Estimator

estimator = Estimator(
    ...
    volume_kms_key='arn:aws:kms:region:account:key/key-id',
    output_kms_key='arn:aws:kms:region:account:key/key-id'
)
```

**EBS Encryption:**
- Training instance volumes
- KMS-encrypted by default
- Cannot disable for SageMaker

**Encryption in Transit:**
- TLS/SSL for all API calls
- HTTPS for data transfer
- Inter-node encryption for distributed training

**SageMaker Network Isolation:**
```python
estimator = Estimator(
    ...
    enable_network_isolation=True  # No internet access
)
```

**Data Access Control:**
- S3 bucket policies
- IAM policies
- VPC endpoints for S3
- Access logging

#### 9.4 Infrastructure Security

**VPC Configuration:**

**Why VPC?**
- Network isolation
- Private communication
- Control traffic
- Compliance requirements

**SageMaker VPC Mode:**
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    ...
    subnets=['subnet-12345', 'subnet-67890'],
    security_group_ids=['sg-12345'],
    enable_network_isolation=True
)
```

**VPC Components:**
- **Subnets**: Isolate resources
- **Security Groups**: Instance-level firewall
- **NACLs**: Subnet-level firewall
- **VPC Endpoints**: Private AWS service access

**VPC Endpoints:**
- **Interface Endpoints**: ENI-based, most services
- **Gateway Endpoints**: S3, DynamoDB
- No internet gateway needed
- Reduced data transfer costs

**Security Groups:**
```python
# Inbound rules
- Allow port 443 (HTTPS) from application layer
- Allow port 2049 (EFS) from SageMaker subnets

# Outbound rules
- Allow all to VPC CIDR (for internal communication)
- Allow 443 to S3 VPC endpoint
- Deny all else
```

**PrivateLink:**
- Private connectivity to services
- No exposure to internet
- Enhanced security

#### 9.5 Secrets Management

**AWS Secrets Manager:**
- Store database credentials
- API keys
- Encryption keys
- Automatic rotation
- Audit access

```python
import boto3
import json

def get_secret():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(
        SecretId='my-database-secret'
    )
    secret = json.loads(response['SecretString'])
    return secret

# Use in SageMaker training
secrets = get_secret()
database_url = f"postgresql://{secrets['username']}:{secrets['password']}@{secrets['host']}"
```

**AWS Systems Manager Parameter Store:**
- Store configuration data
- Secure strings with KMS
- Hierarchical storage
- Version control
- Free tier available

#### 9.6 Logging and Monitoring

**CloudWatch Logs:**
- Training job logs
- Endpoint logs
- Processing job logs
- Retention policies
- Log encryption

**CloudTrail:**
- API call logging
- Governance and compliance
- Security analysis
- Operational troubleshooting

**Important Events to Monitor:**
- Model training start/stop
- Endpoint creation/deletion
- Model deployment
- Data access
- IAM changes
- Configuration changes

**Example CloudTrail Query:**
```sql
-- Find who created endpoints
SELECT 
    userIdentity.principalId,
    eventTime,
    requestParameters.endpointName
FROM cloudtrail_logs
WHERE eventName = 'CreateEndpoint'
    AND eventTime > '2024-01-01'
```

**CloudWatch Metrics:**
- Invocations
- ModelLatency
- Overhead Latency
- CPUUtilization
- MemoryUtilization
- DiskUtilization

**CloudWatch Alarms:**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='HighModelLatency',
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Statistic='Average',
    Period=300,
    EvaluationPeriods=2,
    Threshold=1000,  # milliseconds
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:region:account:topic']
)
```

**SageMaker Model Monitor:**
- Data quality monitoring
- Model quality monitoring
- Bias drift detection
- Feature attribution drift
- Automated alerts

#### 9.7 Compliance and Governance

**Compliance Programs:**
- HIPAA
- PCI DSS
- SOC 1, 2, 3
- ISO 27001
- GDPR
- FedRAMP

**AWS Artifact:**
- Compliance reports
- Agreements
- On-demand access

**Tagging Strategy:**
```python
tags = [
    {'Key': 'Environment', 'Value': 'Production'},
    {'Key': 'Project', 'Value': 'Fraud-Detection'},
    {'Key': 'CostCenter', 'Value': 'ML-Team'},
    {'Key': 'Compliance', 'Value': 'PCI'},
    {'Key': 'DataClassification', 'Value': 'Confidential'}
]

estimator = Estimator(
    ...
    tags=tags
)
```

**Benefits of Tagging:**
- Cost allocation
- Access control
- Automation
- Compliance tracking
- Resource organization

**AWS Config:**
- Resource configuration tracking
- Compliance auditing
- Change notifications
- Configuration history

**Example Config Rule:**
- Ensure all SageMaker endpoints are encrypted
- Check VPC configuration
- Verify logging enabled
- Enforce tagging standards

**Service Control Policies (SCPs):**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": "sagemaker:CreateEndpoint",
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "sagemaker:VpcSecurityGroupIds": "sg-required-group"
                }
            }
        }
    ]
}
```

#### 9.8 Data Privacy and Governance

**Data Classification:**
- Public
- Internal
- Confidential
- Restricted

**PII Handling:**
- Identify PII in datasets
- Mask or tokenize
- Encrypt
- Access controls
- Audit logging

**Amazon Macie:**
- Automated PII discovery
- S3 data classification
- Sensitive data alerts
- Compliance reporting

**Data Lineage:**
- Track data sources
- Transformation history
- Usage tracking
- Compliance documentation

**SageMaker Lineage Tracking:**
```python
from sagemaker.lineage import context, artifact, association

# Create dataset artifact
dataset = artifact.Artifact.create(
    artifact_name='training-dataset',
    artifact_type='Dataset',
    source_uri='s3://bucket/data'
)

# Create training context
training = context.Context.create(
    context_name='model-training',
    context_type='TrainingJob',
    source_uri='arn:aws:sagemaker:...'
)

# Associate dataset with training
association.Association.create(
    source_arn=dataset.artifact_arn,
    destination_arn=training.context_arn,
    association_type='ContributedTo'
)
```

#### 9.9 Model Security

**Model Signing:**
- Verify model integrity
- Prevent tampering
- Use KMS for signing
- Validate before deployment

**Model Access Control:**
- IAM policies for model access
- Model Registry permissions
- Approval workflows
- Version control

**Adversarial Robustness:**
- Input validation
- Anomaly detection
- Rate limiting
- Model monitoring

**Model Theft Prevention:**
- Watermarking
- Model extraction detection
- API rate limiting
- Access logging

#### 9.10 Incident Response

**Preparation:**
- Define security contacts
- Document runbooks
- Establish escalation procedures
- Regular drills

**Detection:**
- CloudWatch alarms
- GuardDuty findings
- Security Hub alerts
- Anomaly detection

**Response:**
1. Identify and contain
2. Investigate and analyze
3. Remediate
4. Document lessons learned
5. Update procedures

**AWS Security Hub:**
- Centralized security view
- Automated compliance checks
- Finding aggregation
- Integration with other services

**Example Response Actions:**
- Disable compromised endpoints
- Rotate credentials
- Revoke IAM permissions
- Isolate affected resources
- Enable additional logging

### Module Summary
- Implement least privilege with IAM roles and policies
- Encrypt data at rest (KMS) and in transit (TLS)
- Deploy in VPC for network isolation
- Use Secrets Manager for credentials
- Enable comprehensive logging (CloudTrail, CloudWatch)
- Implement monitoring and alerting
- Ensure compliance with tagging and AWS Config
- Protect sensitive data with Macie
- Secure models with signing and access control
- Prepare incident response procedures

---

*Continue to Module 10 for MLOps and Automation...*
