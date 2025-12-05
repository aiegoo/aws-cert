// AWS MLA-C01 Exam Cards Data - 65 Questions covering all domains
const cards = [
    // ===== DOMAIN 1: DATA ENGINEERING FOR ML (34% - 22 cards) =====
    
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "S3",
        difficulty: "Easy",
        question: "What S3 storage class should you use for ML training data that's accessed frequently?",
        answer: `<ul>
            <li><strong>S3 Standard:</strong> Best for frequently accessed data</li>
            <li>Low latency and high throughput</li>
            <li>Most expensive but no retrieval fees</li>
            <li><strong>Training data needs:</strong> Fast access during epochs</li>
            <li>Use S3 Intelligent-Tiering if access patterns vary</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "AWS Glue",
        difficulty: "Medium",
        question: "How do you prepare and transform data using AWS Glue for ML training?",
        answer: `<ul>
            <li><strong>Glue Crawlers:</strong> Automatically discover schema and create Data Catalog</li>
            <li><strong>Glue ETL Jobs:</strong> Transform data using PySpark or Python Shell</li>
            <li><strong>Glue DataBrew:</strong> Visual data preparation without code</li>
            <li><strong>Output:</strong> Store in S3 in formats like Parquet, CSV</li>
            <li>Integrates with SageMaker for ML pipelines</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Athena",
        difficulty: "Easy",
        question: "How do you query training data in S3 using Amazon Athena?",
        answer: `<ul>
            <li>Serverless SQL queries directly on S3 data</li>
            <li>Use AWS Glue Data Catalog for schema</li>
            <li>Pay per query (data scanned)</li>
            <li><strong>Example:</strong> <code>SELECT * FROM training_data WHERE label = 'positive'</code></li>
            <li>Export results to S3 for SageMaker input</li>
            <li>Supports Parquet, ORC, JSON, CSV</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/athena/latest/ug/what-is.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Kinesis",
        difficulty: "Medium",
        question: "What's the difference between Kinesis Data Streams and Kinesis Data Firehose for ML data ingestion?",
        answer: `<p><strong>Kinesis Data Streams:</strong></p>
        <ul>
            <li>Real-time streaming (milliseconds)</li>
            <li>Custom processing with Lambda/applications</li>
            <li>Manual scaling (shards)</li>
            <li>Data retention 1-365 days</li>
        </ul>
        <p><strong>Kinesis Data Firehose:</strong></p>
        <ul>
            <li>Near real-time (60 seconds minimum)</li>
            <li>Direct delivery to S3, Redshift, Elasticsearch</li>
            <li>Auto-scaling</li>
            <li>No data retention</li>
            <li><strong>For ML:</strong> Firehose → S3 for batch training</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/kinesis/latest/dev/introduction.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Formats",
        difficulty: "Medium",
        question: "What data format is best for ML training in S3?",
        answer: `<ul>
            <li><strong>Parquet:</strong> Columnar, compressed, best for analytics</li>
            <li><strong>Advantages:</strong> 10x smaller files, faster reads</li>
            <li><strong>CSV:</strong> Human-readable but slower and larger</li>
            <li><strong>RecordIO-Protobuf:</strong> SageMaker optimized format</li>
            <li><strong>TFRecord:</strong> TensorFlow native format</li>
            <li><strong>Best Practice:</strong> Use Parquet for large datasets</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "EMR",
        difficulty: "Hard",
        question: "When should you use Amazon EMR for ML data preprocessing?",
        answer: `<ul>
            <li><strong>Use EMR when:</strong> Processing petabyte-scale data</li>
            <li>Complex Spark/Hadoop transformations needed</li>
            <li>Need distributed computing for feature engineering</li>
            <li><strong>vs Glue:</strong> EMR gives more control, Glue is simpler</li>
            <li>Integrates with SageMaker via S3</li>
            <li>Can use Spark MLlib for feature engineering</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "SageMaker Ground Truth",
        difficulty: "Medium",
        question: "What is Amazon SageMaker Ground Truth?",
        answer: `<ul>
            <li><strong>Purpose:</strong> Create high-quality labeled datasets</li>
            <li>Human workforce options: Amazon Mechanical Turk, private, vendor</li>
            <li><strong>Active Learning:</strong> ML models suggest labels to reduce cost</li>
            <li>Supports: Image, text, video, 3D point cloud labeling</li>
            <li>Built-in workflows for common tasks</li>
            <li>Output: Labeled manifest files for training</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Wrangler",
        difficulty: "Easy",
        question: "What is SageMaker Data Wrangler?",
        answer: `<ul>
            <li>Visual interface for data preparation</li>
            <li>Import from S3, Athena, Redshift</li>
            <li>300+ built-in transformations</li>
            <li>Generate data quality reports</li>
            <li>Export to SageMaker Pipelines or Feature Store</li>
            <li>No code required for common transformations</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Feature Store",
        difficulty: "Medium",
        question: "What is SageMaker Feature Store and when to use it?",
        answer: `<ul>
            <li><strong>Purpose:</strong> Centralized repository for ML features</li>
            <li><strong>Online Store:</strong> Low-latency for real-time inference (DynamoDB)</li>
            <li><strong>Offline Store:</strong> For training (S3)</li>
            <li>Feature versioning and lineage tracking</li>
            <li>Consistent features across training and inference</li>
            <li><strong>Use when:</strong> Reusing features across models</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Pipeline",
        difficulty: "Medium",
        question: "How do you create an end-to-end data pipeline for ML?",
        answer: `<ol>
            <li><strong>Ingestion:</strong> Kinesis/S3/Database → Raw data</li>
            <li><strong>Processing:</strong> Glue/EMR → Transform and clean</li>
            <li><strong>Feature Engineering:</strong> Data Wrangler/EMR</li>
            <li><strong>Storage:</strong> S3 (Parquet format)</li>
            <li><strong>Feature Store:</strong> For reusable features</li>
            <li><strong>Orchestration:</strong> Step Functions or SageMaker Pipelines</li>
        </ol>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Lake Formation",
        difficulty: "Hard",
        question: "What is AWS Lake Formation for ML?",
        answer: `<ul>
            <li>Simplifies building and managing data lakes</li>
            <li>Centralized permissions and governance</li>
            <li>Import from RDS, S3, on-premises databases</li>
            <li>Automatic data cataloging with Glue</li>
            <li>Column and row-level security</li>
            <li><strong>ML Use Case:</strong> Secure access to training data across teams</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "DynamoDB",
        difficulty: "Medium",
        question: "How do you export DynamoDB data for ML training?",
        answer: `<ul>
            <li><strong>DynamoDB → S3 Export:</strong> Point-in-time snapshots</li>
            <li>Export to S3 in DynamoDB JSON or ION format</li>
            <li>Use Glue to convert to training format</li>
            <li><strong>Streams:</strong> Real-time changes to Kinesis → S3</li>
            <li>No impact on table performance</li>
            <li>Encrypted exports supported</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/S3DataExport.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "RDS",
        difficulty: "Easy",
        question: "How do you extract data from RDS for ML training?",
        answer: `<ul>
            <li><strong>AWS Glue:</strong> JDBC connection to RDS</li>
            <li>Create ETL job to export to S3</li>
            <li><strong>Data Pipeline:</strong> Schedule exports</li>
            <li><strong>Lambda:</strong> Trigger exports on events</li>
            <li>Use read replicas to avoid production impact</li>
            <li>Export in Parquet format for efficiency</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/populate-add-connection.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Redshift",
        difficulty: "Medium",
        question: "How do you use Redshift data for ML training?",
        answer: `<ul>
            <li><strong>UNLOAD command:</strong> Export to S3 in Parquet</li>
            <li><strong>Redshift ML:</strong> Train models directly in Redshift</li>
            <li><strong>Spectrum:</strong> Query S3 data from Redshift</li>
            <li>Use for large-scale data warehouse aggregations</li>
            <li>Integrates with SageMaker for feature engineering</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/c_redshift-ml.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Quality",
        difficulty: "Medium",
        question: "How do you ensure data quality for ML training?",
        answer: `<ul>
            <li><strong>Glue Data Quality:</strong> Automated quality checks</li>
            <li><strong>Data Wrangler:</strong> Generate quality reports</li>
            <li><strong>Checks:</strong> Missing values, outliers, distribution shifts</li>
            <li>Set up CloudWatch alarms for data drift</li>
            <li>Validate schema consistency</li>
            <li>Monitor data freshness and completeness</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/glue-data-quality.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Batch Transform",
        difficulty: "Easy",
        question: "What is SageMaker Batch Transform?",
        answer: `<ul>
            <li>Run inference on entire datasets at once</li>
            <li>No need for persistent endpoint</li>
            <li>Input/output from S3</li>
            <li>Cost-effective for large batch processing</li>
            <li>Automatic data splitting across instances</li>
            <li><strong>Use when:</strong> Offline predictions, not real-time</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Processing",
        difficulty: "Hard",
        question: "What's the best practice for handling imbalanced datasets in ML?",
        answer: `<ul>
            <li><strong>Techniques:</strong> SMOTE, undersampling, oversampling</li>
            <li>Use class weights in training</li>
            <li><strong>SageMaker:</strong> Built-in algorithms handle class weights</li>
            <li>Stratified sampling for train/test split</li>
            <li>Consider specialized metrics (F1, AUC-ROC)</li>
            <li>Use SageMaker Processing for data balancing</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Encryption",
        difficulty: "Medium",
        question: "How do you encrypt ML training data?",
        answer: `<ul>
            <li><strong>S3:</strong> SSE-S3, SSE-KMS, or SSE-C encryption</li>
            <li><strong>SageMaker:</strong> Encrypts training data in transit and at rest</li>
            <li>Use KMS keys for encryption control</li>
            <li>Enable encryption on Feature Store</li>
            <li>VPC endpoints for secure data transfer</li>
            <li><strong>Best Practice:</strong> Use SSE-KMS with custom keys</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "SageMaker Processing",
        difficulty: "Medium",
        question: "What is SageMaker Processing and when to use it?",
        answer: `<ul>
            <li>Run data preprocessing and feature engineering jobs</li>
            <li>Use custom containers or built-in frameworks</li>
            <li>Automatically scales compute resources</li>
            <li>Input from S3, output to S3</li>
            <li><strong>Use cases:</strong> Data validation, feature extraction, evaluation</li>
            <li>Integrates with SageMaker Pipelines</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Versioning",
        difficulty: "Hard",
        question: "How do you version datasets for ML reproducibility?",
        answer: `<ul>
            <li><strong>S3 Versioning:</strong> Track changes to training data</li>
            <li><strong>Feature Store:</strong> Automatic feature versioning</li>
            <li>Use SageMaker Experiments to track data versions</li>
            <li>Store metadata in DynamoDB or Data Catalog</li>
            <li>Tag S3 objects with version information</li>
            <li><strong>Best Practice:</strong> Immutable datasets with timestamps</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "MSK",
        difficulty: "Hard",
        question: "When should you use Amazon MSK (Managed Kafka) for ML?",
        answer: `<ul>
            <li><strong>Use for:</strong> High-throughput streaming data</li>
            <li>Complex event processing before ML inference</li>
            <li>Integrates with Kinesis and S3</li>
            <li><strong>vs Kinesis:</strong> MSK better for Kafka ecosystem compatibility</li>
            <li>Stream training data updates in real-time</li>
            <li>Supports exactly-once semantics</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/msk/latest/developerguide/what-is-msk.html"
    },
    {
        domain: "data-engineering",
        domainLabel: "Data Engineering",
        category: "Data Catalog",
        difficulty: "Easy",
        question: "What is AWS Glue Data Catalog?",
        answer: `<ul>
            <li>Centralized metadata repository</li>
            <li>Stores table definitions, schemas, locations</li>
            <li>Used by Athena, Redshift Spectrum, EMR, Glue</li>
            <li>Automatically populated by Glue Crawlers</li>
            <li>Enables data discovery and governance</li>
            <li><strong>ML Use:</strong> Track all training dataset metadata</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html"
    },

    // ===== DOMAIN 2: MODEL DEVELOPMENT (26% - 17 cards) =====
    
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "SageMaker",
        difficulty: "Easy",
        question: "What are the main components of SageMaker for model development?",
        answer: `<ul>
            <li><strong>Studio:</strong> Integrated IDE for ML</li>
            <li><strong>Notebooks:</strong> Jupyter notebooks for development</li>
            <li><strong>Training:</strong> Managed training infrastructure</li>
            <li><strong>Built-in Algorithms:</strong> Pre-built ML algorithms</li>
            <li><strong>Debugger:</strong> Monitor and debug training</li>
            <li><strong>Experiments:</strong> Track and compare runs</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Training",
        difficulty: "Medium",
        question: "What are SageMaker Training Job instance types?",
        answer: `<ul>
            <li><strong>ml.m5:</strong> General purpose, balanced compute/memory</li>
            <li><strong>ml.c5:</strong> Compute optimized for CPU-intensive tasks</li>
            <li><strong>ml.p3/p4:</strong> GPU instances for deep learning</li>
            <li><strong>ml.g4dn:</strong> Cost-effective GPU instances</li>
            <li><strong>Spot Instances:</strong> Up to 90% cost savings</li>
            <li><strong>Best Practice:</strong> Start small, scale up as needed</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Built-in Algorithms",
        difficulty: "Medium",
        question: "What are key SageMaker built-in algorithms?",
        answer: `<ul>
            <li><strong>XGBoost:</strong> Gradient boosting for tabular data</li>
            <li><strong>Linear Learner:</strong> Classification and regression</li>
            <li><strong>Factorization Machines:</strong> Sparse data, recommendations</li>
            <li><strong>DeepAR:</strong> Time series forecasting</li>
            <li><strong>Image Classification:</strong> CNN-based image training</li>
            <li><strong>Object Detection:</strong> Bounding box predictions</li>
            <li><strong>Semantic Segmentation:</strong> Pixel-level classification</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Hyperparameters",
        difficulty: "Hard",
        question: "What is SageMaker Automatic Model Tuning (Hyperparameter Tuning)?",
        answer: `<ul>
            <li>Automated hyperparameter optimization</li>
            <li><strong>Strategies:</strong> Bayesian, Random, Grid search</li>
            <li>Define objective metric to optimize</li>
            <li>Specify hyperparameter ranges</li>
            <li>Parallel training jobs for faster tuning</li>
            <li><strong>Best Practice:</strong> Use Bayesian optimization</li>
            <li>Can reduce tuning time by 10x vs manual</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Debugger",
        difficulty: "Medium",
        question: "What is SageMaker Debugger?",
        answer: `<ul>
            <li>Real-time monitoring of training jobs</li>
            <li>Detects: Overfitting, vanishing gradients, exploding tensors</li>
            <li>Built-in rules for common issues</li>
            <li>Custom rules supported</li>
            <li>Visualize tensors in SageMaker Studio</li>
            <li>Automatically stops training if issues detected</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Distributed Training",
        difficulty: "Hard",
        question: "What distributed training strategies does SageMaker support?",
        answer: `<ul>
            <li><strong>Data Parallelism:</strong> Split data across instances</li>
            <li><strong>Model Parallelism:</strong> Split large models across GPUs</li>
            <li><strong>SageMaker Distributed:</strong> Library for both strategies</li>
            <li>Supports TensorFlow, PyTorch, MXNet</li>
            <li>Automatic sharding and gradient aggregation</li>
            <li><strong>Use when:</strong> Training very large models or datasets</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Experiments",
        difficulty: "Easy",
        question: "What is SageMaker Experiments?",
        answer: `<ul>
            <li>Track and compare ML experiments</li>
            <li>Organize training runs into experiments</li>
            <li>Log parameters, metrics, artifacts</li>
            <li>Compare runs side-by-side</li>
            <li>Visualize metrics over time</li>
            <li>Automatic lineage tracking</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "AutoML",
        difficulty: "Medium",
        question: "What is SageMaker Autopilot?",
        answer: `<ul>
            <li>Automated ML model development</li>
            <li>Automatically explores algorithms and hyperparameters</li>
            <li>Generates notebook showing the process</li>
            <li>Supports: Binary/multiclass classification, regression</li>
            <li>Handles feature engineering automatically</li>
            <li><strong>Use when:</strong> Quick baseline models needed</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Frameworks",
        difficulty: "Easy",
        question: "What ML frameworks does SageMaker support?",
        answer: `<ul>
            <li><strong>Deep Learning:</strong> TensorFlow, PyTorch, MXNet</li>
            <li><strong>Scikit-learn:</strong> Traditional ML algorithms</li>
            <li><strong>Hugging Face:</strong> NLP transformers</li>
            <li><strong>SparkML:</strong> Distributed ML on Spark</li>
            <li><strong>Bring Your Own:</strong> Custom containers</li>
            <li>Pre-built containers with optimized versions</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Checkpoints",
        difficulty: "Medium",
        question: "How do you handle training checkpoints in SageMaker?",
        answer: `<ul>
            <li>Save checkpoints to S3 during training</li>
            <li>Resume training from last checkpoint</li>
            <li>Useful for Spot instances (can be interrupted)</li>
            <li>Configure checkpoint frequency</li>
            <li>SageMaker manages checkpoint lifecycle</li>
            <li><strong>Best Practice:</strong> Always enable for long training jobs</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Managed Spot",
        difficulty: "Medium",
        question: "How do Managed Spot Training Instances work?",
        answer: `<ul>
            <li>Up to 90% cost savings vs on-demand</li>
            <li>Can be interrupted if capacity needed</li>
            <li><strong>Requirements:</strong> Must support checkpointing</li>
            <li>SageMaker automatically restarts on interruption</li>
            <li>Set maximum wait time for spot capacity</li>
            <li><strong>Best for:</strong> Fault-tolerant, long-running jobs</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Model Registry",
        difficulty: "Easy",
        question: "What is SageMaker Model Registry?",
        answer: `<ul>
            <li>Centralized repository for model versions</li>
            <li>Track model lineage and metadata</li>
            <li>Approval workflows for production deployment</li>
            <li>Version models with status (Pending, Approved, Rejected)</li>
            <li>Integrate with CI/CD pipelines</li>
            <li>Audit trail for compliance</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Transfer Learning",
        difficulty: "Medium",
        question: "How do you implement transfer learning in SageMaker?",
        answer: `<ul>
            <li>Use pre-trained models from Model Zoo</li>
            <li><strong>Built-in algorithms:</strong> Image Classification supports transfer learning</li>
            <li>Load pre-trained weights from S3</li>
            <li>Fine-tune on your specific dataset</li>
            <li>Freeze early layers, train final layers</li>
            <li><strong>Reduces:</strong> Training time and data requirements</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Clarify",
        difficulty: "Hard",
        question: "What is SageMaker Clarify?",
        answer: `<ul>
            <li>Detect bias in training data and models</li>
            <li>Explain model predictions (SHAP values)</li>
            <li><strong>Bias Metrics:</strong> Class imbalance, demographic parity</li>
            <li>Generate bias reports pre and post-training</li>
            <li>Feature importance analysis</li>
            <li><strong>Required for:</strong> Responsible AI practices</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Local Mode",
        difficulty: "Easy",
        question: "What is SageMaker Local Mode?",
        answer: `<ul>
            <li>Test training and inference locally</li>
            <li>Uses Docker containers on your machine</li>
            <li>Faster iteration during development</li>
            <li>Same code works locally and on SageMaker</li>
            <li>Debug before launching expensive training</li>
            <li><strong>Use instance type:</strong> <code>local</code> or <code>local_gpu</code></li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/local-mode.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "Training Metrics",
        difficulty: "Medium",
        question: "How do you monitor training metrics in SageMaker?",
        answer: `<ul>
            <li>Emit metrics from training script</li>
            <li>SageMaker automatically captures to CloudWatch</li>
            <li>View metrics in SageMaker Studio</li>
            <li>Define custom metrics with regex patterns</li>
            <li>Set CloudWatch alarms on metrics</li>
            <li><strong>Common metrics:</strong> Loss, accuracy, validation score</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html"
    },
    {
        domain: "model-development",
        domainLabel: "Model Development",
        category: "JumpStart",
        difficulty: "Easy",
        question: "What is SageMaker JumpStart?",
        answer: `<ul>
            <li>Pre-trained models and solution templates</li>
            <li>One-click deployment of popular models</li>
            <li>Includes: Computer vision, NLP, tabular models</li>
            <li>Foundation models (LLMs) available</li>
            <li>Fine-tune on your data with few clicks</li>
            <li><strong>Use for:</strong> Quick model prototypes</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html"
    },

    // ===== DOMAIN 3: DEPLOYMENT & MLOPS (22% - 15 cards) =====
    
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Endpoints",
        difficulty: "Easy",
        question: "What is a SageMaker real-time endpoint?",
        answer: `<ul>
            <li>Persistent HTTPS endpoint for real-time predictions</li>
            <li>Auto-scaling based on traffic</li>
            <li>Low-latency inference (milliseconds)</li>
            <li>Supports A/B testing via multiple variants</li>
            <li>Deploy multiple models behind one endpoint</li>
            <li><strong>Billing:</strong> Pay for instance runtime</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Serverless Inference",
        difficulty: "Medium",
        question: "What is SageMaker Serverless Inference?",
        answer: `<ul>
            <li>No need to provision or manage servers</li>
            <li>Automatically scales to zero when not in use</li>
            <li>Pay only for inference time</li>
            <li><strong>Best for:</strong> Intermittent traffic, cost optimization</li>
            <li>Cold start latency (seconds)</li>
            <li><strong>vs Real-time:</strong> Lower cost but higher latency</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Model Monitor",
        difficulty: "Medium",
        question: "What is SageMaker Model Monitor?",
        answer: `<ul>
            <li>Continuous monitoring of deployed models</li>
            <li><strong>Detects:</strong> Data drift, model quality degradation, bias drift</li>
            <li>Compare against baseline dataset</li>
            <li>Automatic CloudWatch alarms on violations</li>
            <li>Capture inference data for analysis</li>
            <li><strong>Schedules:</strong> Hourly, daily, or custom</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Pipelines",
        difficulty: "Hard",
        question: "What is SageMaker Pipelines?",
        answer: `<ul>
            <li>CI/CD orchestration for ML workflows</li>
            <li>Define pipeline steps: Processing, Training, Evaluation, Deployment</li>
            <li>JSON/Python SDK for pipeline definition</li>
            <li>Automatic lineage tracking</li>
            <li>Conditional execution based on metrics</li>
            <li><strong>Integrates:</strong> Model Registry, Feature Store, Experiments</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Multi-Model Endpoints",
        difficulty: "Hard",
        question: "What are SageMaker Multi-Model Endpoints?",
        answer: `<ul>
            <li>Host multiple models behind single endpoint</li>
            <li>Models loaded dynamically from S3</li>
            <li><strong>Cost savings:</strong> Share compute across models</li>
            <li>Best for: Hundreds of similar models</li>
            <li>Models share container and instances</li>
            <li><strong>Example use:</strong> Personalized models per customer</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Inference Pipelines",
        difficulty: "Medium",
        question: "What are SageMaker Inference Pipelines?",
        answer: `<ul>
            <li>Chain 2-15 containers for inference</li>
            <li>Combine preprocessing + model + postprocessing</li>
            <li>Deploy entire pipeline as single endpoint</li>
            <li>Supports Spark ML and Scikit-learn containers</li>
            <li><strong>Use case:</strong> Feature engineering + prediction</li>
            <li>Reduces inference latency vs separate calls</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Shadow Tests",
        difficulty: "Hard",
        question: "What is Shadow Testing in SageMaker?",
        answer: `<ul>
            <li>Test new model without affecting production</li>
            <li>Route copy of production traffic to shadow variant</li>
            <li>Compare predictions without serving to users</li>
            <li>Validate model before full deployment</li>
            <li>Monitor both variants in parallel</li>
            <li><strong>Safer than:</strong> Direct production replacement</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-shadow-deployment.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Blue/Green Deployment",
        difficulty: "Medium",
        question: "How do you implement Blue/Green deployment in SageMaker?",
        answer: `<ul>
            <li>Create new endpoint with updated model (Green)</li>
            <li>Test green endpoint thoroughly</li>
            <li>Switch traffic from blue to green</li>
            <li>Keep blue for rollback if needed</li>
            <li>Can use endpoint variants for gradual shift</li>
            <li><strong>Zero downtime</strong> deployment strategy</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Auto Scaling",
        difficulty: "Medium",
        question: "How does SageMaker endpoint auto-scaling work?",
        answer: `<ul>
            <li>Based on target tracking or step scaling</li>
            <li><strong>Metric:</strong> <code>SageMakerVariantInvocationsPerInstance</code></li>
            <li>Set min/max instance counts</li>
            <li>Scale up: When metric exceeds target</li>
            <li>Scale down: Gradual after cooldown period</li>
            <li><strong>Best Practice:</strong> Set target to 70% of max capacity</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Edge Deployment",
        difficulty: "Hard",
        question: "What is SageMaker Edge Manager?",
        answer: `<ul>
            <li>Deploy models to edge devices (IoT, mobile)</li>
            <li>Optimize models for edge hardware</li>
            <li>Monitor edge model performance</li>
            <li>Update models on edge devices remotely</li>
            <li>Collect predictions for retraining</li>
            <li><strong>Use with:</strong> SageMaker Neo for optimization</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/edge.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Neo",
        difficulty: "Medium",
        question: "What is SageMaker Neo?",
        answer: `<ul>
            <li>Optimize models for deployment</li>
            <li>Compile models for specific hardware</li>
            <li>Supports: TensorFlow, PyTorch, MXNet, ONNX</li>
            <li><strong>Benefits:</strong> Up to 2x faster inference, smaller size</li>
            <li>Works with: Cloud, edge, IoT devices</li>
            <li>No code changes required</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Elastic Inference",
        difficulty: "Medium",
        question: "What is Amazon Elastic Inference (EI)?",
        answer: `<ul>
            <li>Attach low-cost GPU acceleration to instances</li>
            <li>Fractional GPU capacity (not full GPU)</li>
            <li>Cost-effective for inference workloads</li>
            <li><strong>Note:</strong> Being replaced by Inferentia/Graviton</li>
            <li>Best for: Deep learning inference optimization</li>
            <li>Works with SageMaker endpoints and EC2</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/elastic-inference/latest/developerguide/what-is-ei.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Model Explainability",
        difficulty: "Hard",
        question: "How do you implement model explainability in production?",
        answer: `<ul>
            <li><strong>SageMaker Clarify:</strong> SHAP values for feature importance</li>
            <li>Enable explainability on endpoints</li>
            <li>Generate explanations per prediction</li>
            <li>Monitor explanation drift over time</li>
            <li>Required for regulated industries</li>
            <li><strong>Trade-off:</strong> Adds latency to inference</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Model Retraining",
        difficulty: "Medium",
        question: "What are strategies for automated model retraining?",
        answer: `<ul>
            <li><strong>Scheduled:</strong> EventBridge triggers SageMaker Pipeline</li>
            <li><strong>Data-driven:</strong> Retrain when new data threshold met</li>
            <li><strong>Performance-driven:</strong> Model Monitor detects drift</li>
            <li>Use SageMaker Pipelines for automation</li>
            <li>A/B test new model vs current</li>
            <li><strong>Best Practice:</strong> Combine scheduled + drift detection</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-schedule-expression.html"
    },
    {
        domain: "deployment",
        domainLabel: "Deployment & MLOps",
        category: "Deployment Guards",
        difficulty: "Hard",
        question: "What are SageMaker Deployment Guardrails?",
        answer: `<ul>
            <li>Automated rollback on deployment failures</li>
            <li>Traffic shifting strategies (canary, linear, all-at-once)</li>
            <li>CloudWatch alarms as deployment gates</li>
            <li><strong>Canary:</strong> Route small % of traffic first</li>
            <li><strong>Linear:</strong> Gradually increase traffic percentage</li>
            <li>Auto-rollback if alarms trigger</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html"
    },

    // ===== DOMAIN 4: ML SOLUTION DESIGN (18% - 11 cards) =====
    
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Problem Framing",
        difficulty: "Easy",
        question: "What are the main ML problem types?",
        answer: `<ul>
            <li><strong>Supervised:</strong> Labeled data (classification, regression)</li>
            <li><strong>Unsupervised:</strong> No labels (clustering, dimensionality reduction)</li>
            <li><strong>Reinforcement:</strong> Learn from rewards/penalties</li>
            <li><strong>Semi-supervised:</strong> Mix of labeled and unlabeled</li>
            <li><strong>Classification:</strong> Predict categories</li>
            <li><strong>Regression:</strong> Predict continuous values</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Metrics",
        difficulty: "Medium",
        question: "What evaluation metrics should you use for classification?",
        answer: `<ul>
            <li><strong>Accuracy:</strong> Overall correct predictions</li>
            <li><strong>Precision:</strong> True positives / (TP + FP)</li>
            <li><strong>Recall:</strong> True positives / (TP + FN)</li>
            <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
            <li><strong>AUC-ROC:</strong> Area under ROC curve</li>
            <li><strong>Imbalanced data:</strong> Use F1, AUC instead of accuracy</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Cost Optimization",
        difficulty: "Medium",
        question: "How do you optimize ML costs in AWS?",
        answer: `<ul>
            <li><strong>Spot Instances:</strong> 90% savings for training</li>
            <li><strong>Serverless Inference:</strong> Pay per use, not always-on</li>
            <li><strong>Multi-Model Endpoints:</strong> Share resources</li>
            <li><strong>S3 Intelligent-Tiering:</strong> Auto-move old data</li>
            <li><strong>Right-size instances:</strong> Don't over-provision</li>
            <li><strong>Batch Transform:</strong> vs persistent endpoints</li>
        </ul>`,
        link: "https://aws.amazon.com/sagemaker/pricing/"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Architecture",
        difficulty: "Hard",
        question: "How do you design a real-time recommendation system on AWS?",
        answer: `<ol>
            <li><strong>Data:</strong> DynamoDB for user/item features</li>
            <li><strong>Training:</strong> SageMaker Factorization Machines</li>
            <li><strong>Features:</strong> Feature Store for online features</li>
            <li><strong>Inference:</strong> Real-time endpoint with auto-scaling</li>
            <li><strong>Caching:</strong> ElastiCache for frequent recommendations</li>
            <li><strong>API:</strong> API Gateway + Lambda for requests</li>
        </ol>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "NLP Solutions",
        difficulty: "Medium",
        question: "What AWS services are used for NLP solutions?",
        answer: `<ul>
            <li><strong>SageMaker:</strong> Train custom NLP models (BERT, transformers)</li>
            <li><strong>Comprehend:</strong> Pre-trained sentiment, entity extraction</li>
            <li><strong>Translate:</strong> Neural machine translation</li>
            <li><strong>Transcribe:</strong> Speech-to-text</li>
            <li><strong>Polly:</strong> Text-to-speech</li>
            <li><strong>Bedrock:</strong> Foundation models for text generation</li>
        </ul>`,
        link: "https://aws.amazon.com/machine-learning/ml-use-cases/natural-language-processing/"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Computer Vision",
        difficulty: "Medium",
        question: "What AWS services are used for computer vision?",
        answer: `<ul>
            <li><strong>SageMaker:</strong> Custom image models, object detection</li>
            <li><strong>Rekognition:</strong> Pre-trained face, object, text detection</li>
            <li><strong>Panorama:</strong> Computer vision at edge</li>
            <li><strong>Lookout for Vision:</strong> Defect detection</li>
            <li><strong>Textract:</strong> Document text and table extraction</li>
            <li>Choose based on: Custom vs pre-trained needs</li>
        </ul>`,
        link: "https://aws.amazon.com/machine-learning/ml-use-cases/computer-vision/"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Time Series",
        difficulty: "Hard",
        question: "How do you implement time series forecasting on AWS?",
        answer: `<ul>
            <li><strong>SageMaker DeepAR:</strong> Deep learning for forecasting</li>
            <li><strong>Forecast:</strong> Managed service, no ML expertise needed</li>
            <li><strong>Data:</strong> Historical time series in S3</li>
            <li><strong>Features:</strong> Related time series, static features</li>
            <li><strong>Evaluation:</strong> Backtest on holdout period</li>
            <li><strong>Choose DeepAR:</strong> More flexibility, custom models</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Anomaly Detection",
        difficulty: "Medium",
        question: "What are approaches for anomaly detection in AWS?",
        answer: `<ul>
            <li><strong>SageMaker Random Cut Forest:</strong> Unsupervised anomaly detection</li>
            <li><strong>Lookout for Metrics:</strong> Automated anomaly detection service</li>
            <li><strong>CloudWatch Anomaly Detection:</strong> For metrics</li>
            <li><strong>Approach:</strong> Train on normal data, flag outliers</li>
            <li>Use for: Fraud detection, system monitoring</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Model Selection",
        difficulty: "Hard",
        question: "How do you choose the right ML algorithm?",
        answer: `<ul>
            <li><strong>Tabular data:</strong> XGBoost, Linear Learner</li>
            <li><strong>Images:</strong> CNN-based (Image Classification, Object Detection)</li>
            <li><strong>Text:</strong> Transformers, BlazingText</li>
            <li><strong>Time series:</strong> DeepAR</li>
            <li><strong>Recommendations:</strong> Factorization Machines</li>
            <li><strong>Start simple:</strong> Baseline with Linear Learner</li>
            <li><strong>Iterate:</strong> Try multiple algorithms, compare metrics</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Security",
        difficulty: "Medium",
        question: "How do you secure ML solutions in AWS?",
        answer: `<ul>
            <li><strong>VPC:</strong> Run SageMaker in private VPC</li>
            <li><strong>IAM:</strong> Least privilege roles for SageMaker</li>
            <li><strong>Encryption:</strong> KMS for data at rest and in transit</li>
            <li><strong>Network isolation:</strong> No internet for training jobs</li>
            <li><strong>Secrets Manager:</strong> Store API keys, credentials</li>
            <li><strong>CloudTrail:</strong> Audit all SageMaker API calls</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/security.html"
    },
    {
        domain: "ml-solutions",
        domainLabel: "ML Solution Design",
        category: "Responsible AI",
        difficulty: "Hard",
        question: "How do you implement responsible AI practices?",
        answer: `<ul>
            <li><strong>SageMaker Clarify:</strong> Detect and mitigate bias</li>
            <li>Ensure diverse, representative training data</li>
            <li>Monitor for fairness across demographic groups</li>
            <li>Provide model explanations (SHAP)</li>
            <li>Document model cards with limitations</li>
            <li>Regular bias audits on production models</li>
            <li><strong>Governance:</strong> Model Registry approval workflows</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html"
    }
];
