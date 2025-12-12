// AWS Storage & Lakehouse Architecture Flashcards
const cards = [
    // ===== REDSHIFT SPECTRUM =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Redshift Spectrum",
        difficulty: "Easy",
        question: "Your company stores raw clickstream data in S3. Analysts must query it without loading into Redshift. What is the most cost-effective option?",
        answer: `<p><strong>Choose:</strong> Amazon Redshift Spectrum.</p>
        <ul>
            <li>Queries S3 data in place via external tables</li>
            <li>No COPY/ETL cost; pay only per TB scanned</li>
            <li>Perfect for cold datasets left in S3</li>
            <li>Leverages Redshift compute power for complex queries</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/c-spectrum.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Redshift Spectrum",
        difficulty: "Medium",
        question: "You want Redshift Spectrum to scan S3 data. What configuration step comes first?",
        answer: `<ul>
            <li>Create an external schema pointing to the Glue Data Catalog</li>
            <li>Example: <code>CREATE EXTERNAL SCHEMA spectrum_schema FROM DATA CATALOG</code></li>
            <li>Reference the catalog DB and an IAM role with S3 + Glue permissions</li>
            <li>Only then can Spectrum run SQL on S3 objects</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_SCHEMA.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Redshift Spectrum",
        difficulty: "Easy",
        question: "How can a Redshift cluster join warehouse tables with huge S3 datasets without copying data?",
        answer: `<p><strong>Use:</strong> Amazon Redshift Spectrum.</p>
        <ul>
            <li>Define external tables over S3 objects</li>
            <li>Join them with internal tables in a single SQL statement</li>
            <li>Keeps hot data in Redshift, cold data in S3</li>
            <li>Optimizes cost by tiering storage</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/c-data-processing.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Redshift Spectrum",
        difficulty: "Easy",
        question: "Which statement about Redshift Spectrum is TRUE?",
        answer: `<ul>
            <li>It is a <strong>read-only</strong> SQL capability for querying S3</li>
            <li>Cannot write back to S3 or create indexes</li>
            <li>Uses Redshift compute, with metadata sourced from Glue</li>
            <li>Queries are billed separately based on data scanned</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/c-spectrum-reqs-prereqs.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Redshift Spectrum",
        difficulty: "Medium",
        question: "A Lakehouse must support near-real-time dashboards over S3 while analysts already use Redshift. Which service gives low-latency SQL?",
        answer: `<p><strong>Answer:</strong> Redshift Spectrum (extend existing BI to S3).</p>
        <ul>
            <li>Leverages cluster compute for fast joins across hot/cold data</li>
            <li>No extra service to manage when Redshift already powers BI</li>
            <li>Athena is a fallback when no Redshift cluster exists</li>
            <li>Better performance for complex joins and aggregations</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/dg/c-spectrum-what-is.html"
    },

    // ===== LAKE FORMATION =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Easy",
        question: "You need a single permission model for column-level access across Athena and Redshift Spectrum. Which service provides that?",
        answer: `<p><strong>Choose:</strong> AWS Lake Formation.</p>
        <ul>
            <li>Centralizes table/column/row permissions</li>
            <li>Applies uniformly via the Glue Data Catalog</li>
            <li>Replaces scattered IAM and bucket policies</li>
            <li>Provides consistent security across analytics services</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Easy",
        question: "When is AWS Lake Formation explicitly required in a Lakehouse design?",
        answer: `<ul>
            <li>When you need centralized governance for S3-backed analytic tables</li>
            <li>Provides column/row-level controls for Athena, Spectrum, EMR</li>
            <li>Unified auditing and permission workflows</li>
            <li>Simplifies compliance and data access management</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/security-data-lake.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Medium",
        question: "A financial firm needs auditable column- and row-level restrictions for S3 data. What should they implement?",
        answer: `<p><strong>Answer:</strong> Lake Formation permissions.</p>
        <ul>
            <li>Enforce fine-grained grants plus CloudTrail logging</li>
            <li>Works consistently across multiple analytics services</li>
            <li>Scales better than DIY IAM policy sprawl</li>
            <li>Provides audit trail for compliance requirements</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/column-level.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Medium",
        question: "Analysts must only see rows where region = 'APAC'. How do you enforce that?",
        answer: `<p><strong>Use:</strong> Lake Formation row-level filters.</p>
        <ul>
            <li>Create row filters or LF-tags that enforce predicates</li>
            <li>Applies automatically in Athena and Spectrum queries</li>
            <li>No code changes needed by analysts</li>
            <li>Centrally managed security policy</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/row-level.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Medium",
        question: "A Lake Formation admin must allow analysts only curated data, not raw zones. What permissions should be granted?",
        answer: `<ul>
            <li>Grant <em>Describe</em> and <em>Select</em> on curated Glue databases/tables only</li>
            <li>Omit grants on raw databases so they are invisible</li>
            <li>Use LF-tags to keep access scoped and maintainable</li>
            <li>Implement principle of least privilege</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/grant-data-permissions.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Lake Formation",
        difficulty: "Medium",
        question: "A data scientist requires cross-account access with audit logs on S3 analytics datasets. What combination delivers this?",
        answer: `<p><strong>Answer:</strong> Lake Formation + Glue Data Catalog.</p>
        <ul>
            <li>Grant LF permissions to external principals</li>
            <li>Policies enforced across Athena/Spectrum/EMR</li>
            <li>CloudTrail logs provide an audit trail</li>
            <li>Supports resource sharing via AWS RAM</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/lake-formation/latest/dg/security-cross-account.html"
    },

    // ===== DATA FORMATS =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Data Formats",
        difficulty: "Easy",
        question: "Which file format gives the best performance/cost balance for long-term S3 analytics?",
        answer: `<p><strong>Choose:</strong> Apache Parquet.</p>
        <ul>
            <li>Columnar + compressed ⇒ fewer bytes scanned</li>
            <li>Supports predicate pushdown in Athena/Spectrum</li>
            <li>Open format for Lakehouse interoperability</li>
            <li>10x smaller files than CSV, faster query performance</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Data Formats",
        difficulty: "Easy",
        question: "What is the primary benefit of Parquet files for Lakehouse analytics?",
        answer: `<ul>
            <li>Columnar compression allows selective reads</li>
            <li>Reduces scanned bytes (lower Athena/Spectrum cost)</li>
            <li>Widely supported across AWS analytics services</li>
            <li>Efficient storage and query performance</li>
        </ul>`,
        link: "https://parquet.apache.org/documentation/latest/"
    },

    // ===== GLUE DATA CATALOG =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Glue Data Catalog",
        difficulty: "Medium",
        question: "Which catalog lets Athena, Spectrum, and EMR share the same S3 table metadata?",
        answer: `<p><strong>Choose:</strong> AWS Glue Data Catalog.</p>
        <ul>
            <li>Central schema registry for S3-backed tables</li>
            <li>Referenced by multiple analytics engines</li>
            <li>Populated via crawlers or ETL jobs</li>
            <li>Hive-compatible metastore</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/populate-data-catalog.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Glue Data Catalog",
        difficulty: "Easy",
        question: "Where is metadata for S3 external tables stored in an AWS Lakehouse?",
        answer: `<p><strong>Answer:</strong> AWS Glue Data Catalog.</p>
        <ul>
            <li>Stores databases, tables, partitions, schemas</li>
            <li>Lake Formation relies on it for governance</li>
            <li>Decouples metadata from any single compute engine</li>
            <li>Fully managed and serverless</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/components-overview.html"
    },
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Glue Data Catalog",
        difficulty: "Easy",
        question: "Why can Athena and Redshift Spectrum query the same external tables?",
        answer: `<ul>
            <li>Both reference the same Glue Data Catalog definitions</li>
            <li>Schema updates become instantly visible to both services</li>
            <li>Lake Formation policies apply uniformly</li>
            <li>Single source of truth for metadata</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/athena/latest/ug/glue-best-practices.html"
    },

    // ===== GLUE CRAWLERS =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Glue",
        difficulty: "Easy",
        question: "Which AWS service scans S3 and keeps the Glue Data Catalog schemas up to date automatically?",
        answer: `<p><strong>Answer:</strong> AWS Glue Crawlers.</p>
        <ul>
            <li>Discover partitions, data types, and table definitions</li>
            <li>Schedule crawls to keep schemas current</li>
            <li>Feed Lake Formation for permission enforcement</li>
            <li>Support incremental crawling for efficiency</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/glue/latest/dg/add-crawler.html"
    },

    // ===== DATA LAYOUT =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Data Layout",
        difficulty: "Medium",
        question: "What S3 layout delivers petabyte-scale performance at the lowest cost?",
        answer: `<p><strong>Best practice:</strong> Large Parquet files organized in partitioned prefixes (e.g., <code>dt=YYYY-MM-DD/</code>).</p>
        <ul>
            <li>Columnar compression minimizes scanned bytes</li>
            <li>Partition pruning skips irrelevant folders</li>
            <li>Fewer, larger files reduce request overhead</li>
            <li>Optimal file size: 128MB - 1GB per file</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/athena/latest/ug/partitions.html"
    },

    // ===== TABLE FORMATS =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Table Formats",
        difficulty: "Medium",
        question: "Your ETL must update small portions of large S3 tables with ACID guarantees. Which pattern solves this?",
        answer: `<p><strong>Use:</strong> ACID table formats such as Apache Iceberg on S3.</p>
        <ul>
            <li>Supports MERGE/UPSERT semantics</li>
            <li>Keeps snapshots for time-travel queries</li>
            <li>Integrates with Athena, EMR, Glue, Redshift (preview)</li>
            <li>Alternative options: Delta Lake, Apache Hudi</li>
        </ul>`,
        link: "https://iceberg.apache.org/"
    },

    // ===== COST OPTIMIZATION =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Cost Optimization",
        difficulty: "Medium",
        question: "How do you cut Redshift costs for historical data while keeping it queryable?",
        answer: `<p><strong>Answer:</strong> Offload cold partitions to S3 and query via Spectrum.</p>
        <ul>
            <li>Keep only hot data on RA3 managed storage</li>
            <li>Spectrum external tables read S3-held history</li>
            <li>Maintains analytics continuity with less cluster storage</li>
            <li>Use S3 Intelligent-Tiering or Glacier for long-term storage</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/redshift/latest/mgmt/c-using-spectrum.html"
    },

    // ===== DATA PROCESSING =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Data Processing Tools",
        difficulty: "Medium",
        question: "Map the right AWS service to each Lakehouse data-processing need: automated ETL, event-driven workflows, orchestration, serverless execution, metadata management.",
        answer: `<ul>
            <li><strong>Automated ETL:</strong> AWS Glue (serverless Spark, crawlers, ETL jobs)</li>
            <li><strong>Event-driven workflows:</strong> Amazon EventBridge (central event bus)</li>
            <li><strong>Workflow orchestration:</strong> AWS Step Functions or AWS MWAA (managed Airflow)</li>
            <li><strong>Serverless execution:</strong> AWS Lambda for lightweight tasks</li>
            <li><strong>Metadata management:</strong> AWS Glue Data Catalog</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/whitepapers/latest/building-data-lakes/lakehouse.html"
    },

    // ===== CACHING =====
    
    {
        domain: "storage",
        domainLabel: "Storage & Data Lakes",
        category: "Caching",
        difficulty: "Easy",
        question: "Which AWS-managed caching service should you highlight in Lakehouse architectures when low-latency access is required?",
        answer: `<p><strong>Answer:</strong> Amazon ElastiCache (Redis or Memcached).</p>
        <ul>
            <li>Provides in-memory caching to offload databases or speed API responses</li>
            <li>Can cache query results or hot feature-store data</li>
            <li>Fully managed with scaling, patching, and monitoring</li>
            <li>Sub-millisecond latency for cached data</li>
        </ul>`,
        link: "https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html"
    }
];
