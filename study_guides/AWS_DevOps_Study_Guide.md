# AWS DevOps Study Guide
*Generated from Korean AWS DevOps Training Materials*

## Overview

This comprehensive guide covers AWS DevOps practices, tools, and methodologies based on official AWS training materials. The content focuses on implementing DevOps principles on AWS infrastructure.

---

## Table of Contents

1. [Introduction to AWS DevOps](#introduction)
2. [IAM (Identity and Access Management)](#iam)
3. [CI/CD Pipeline](#cicd)
4. [Infrastructure as Code](#iac)
5. [Monitoring and Logging](#monitoring)
6. [Key AWS Services for DevOps](#services)
7. [Best Practices](#best-practices)
8. [Exam Preparation](#exam-prep)

---

## Introduction to AWS DevOps

### What is DevOps?

DevOps is a combination of cultural philosophies, practices, and tools that increases an organization's ability to deliver applications and services at high velocity.

### AWS DevOps Benefits

- **Automation**: Automate manual processes
- **Rapid Delivery**: Deploy faster and more frequently
- **Reliability**: Ensure quality through automated testing
- **Scale**: Manage infrastructure at scale
- **Collaboration**: Improve team collaboration

---

## IAM (Identity and Access Management)

### Core Concepts

**IAM Users**: Individual identities with specific credentials
- Each user has unique credentials (username/password or access keys)
- Users can be organized into groups
- Direct policy attachment possible

**IAM Groups**: Collections of users
- Simplifies permission management
- Users inherit group permissions
- Cannot be nested

**IAM Roles**: Temporary credentials for services and users
- Used by AWS services (EC2, Lambda, etc.)
- Cross-account access
- Federated user access
- No long-term credentials

**IAM Policies**: JSON documents defining permissions
- Effect: Allow or Deny
- Action: What can be done
- Resource: Which resources
- Condition: When it applies

### Best Practices

1. **Use Roles Instead of Users for Applications**
   - EC2 instances should use IAM roles
   - Lambda functions use execution roles
   - No hard-coded credentials

2. **Follow Least Privilege Principle**
   - Grant only necessary permissions
   - Start with minimal access
   - Add permissions as needed

3. **Enable MFA for Privileged Users**
   - Root account must have MFA
   - Admin users should have MFA
   - Use hardware or virtual MFA devices

4. **Rotate Credentials Regularly**
   - Access keys should be rotated
   - Passwords should expire
   - Audit unused credentials

---

## CI/CD Pipeline

### AWS CodePipeline

**Purpose**: Orchestrates the entire release process

**Stages**:
1. **Source**: Code repository (CodeCommit, GitHub, S3)
2. **Build**: Compile and test (CodeBuild)
3. **Test**: Automated testing
4. **Deploy**: Deployment to environments (CodeDeploy)

### AWS CodeBuild

**Features**:
- Fully managed build service
- Compiles source code
- Runs tests
- Produces deployable artifacts

**BuildSpec.yml Structure**:
```yaml
version: 0.2
phases:
  install:
    commands:
      - npm install
  build:
    commands:
      - npm run build
  post_build:
    commands:
      - npm test
artifacts:
  files:
    - '**/*'
```

### AWS CodeDeploy

**Deployment Types**:
- **In-place**: Updates existing instances
- **Blue/Green**: Creates new instances, shifts traffic

**Deployment Strategies**:
- **All at once**: Fastest, highest risk
- **Rolling**: Gradual, maintains capacity
- **Canary**: Test with small percentage first
- **Linear**: Gradual percentage increase

---

## Infrastructure as Code

### AWS CloudFormation

**Benefits**:
- Version control for infrastructure
- Repeatable deployments
- Rollback capabilities
- Infrastructure documentation

**Key Concepts**:
- **Templates**: JSON/YAML files describing resources
- **Stacks**: Collection of AWS resources
- **Change Sets**: Preview changes before execution

**Template Structure**:
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Example template'

Resources:
  MyEC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: t2.micro
      
Outputs:
  InstanceId:
    Value: !Ref MyEC2Instance
```

### AWS CDK (Cloud Development Kit)

**Advantages**:
- Use familiar programming languages (Python, TypeScript, Java)
- Object-oriented approach
- Built-in best practices
- Automatic CloudFormation generation

---

## Monitoring and Logging

### Amazon CloudWatch

**Metrics**:
- Default metrics (CPU, Network, Disk)
- Custom metrics
- Alarms based on thresholds

**Logs**:
- Centralized log management
- Log groups and streams
- Retention policies
- Metric filters

**CloudWatch Events/EventBridge**:
- Event-driven automation
- Scheduled events (cron)
- React to AWS service events

### AWS X-Ray

**Purpose**: Distributed tracing for microservices

**Features**:
- Request tracking
- Service map visualization
- Performance analysis
- Error detection

---

## Key AWS Services for DevOps

### Version Control
- **CodeCommit**: Managed Git repository
- Integration with other AWS services
- Secure and scalable

### Compute
- **EC2**: Virtual servers
- **Lambda**: Serverless compute
- **ECS/EKS**: Container orchestration

### Database
- **RDS**: Managed relational databases
- **DynamoDB**: NoSQL database
- **ElastiCache**: In-memory caching

### Networking
- **VPC**: Isolated network environment
- **ELB**: Load balancing
- **Route 53**: DNS service
- **CloudFront**: CDN

### Security
- **IAM**: Identity management
- **KMS**: Key management
- **Secrets Manager**: Credentials management
- **WAF**: Web application firewall

---

## Best Practices

### 1. Automate Everything
- Use Infrastructure as Code
- Automate testing
- Automate deployments
- Automate monitoring and alerts

### 2. Version Control All Code
- Application code
- Infrastructure code
- Configuration files
- Documentation

### 3. Implement Proper Testing
- Unit tests
- Integration tests
- Security scans
- Performance tests

### 4. Use Immutable Infrastructure
- No manual changes to production
- Replace rather than update
- Consistent environments

### 5. Monitor and Log Everything
- Application metrics
- Infrastructure metrics
- Security events
- Performance data

### 6. Implement Security at Every Layer
- Network security (VPC, Security Groups)
- Application security (WAF, encryption)
- Data security (encryption at rest/transit)
- Access control (IAM)

### 7. Design for Failure
- Multi-AZ deployments
- Auto-scaling
- Health checks
- Automated recovery

---

## Exam Preparation Tips

### Key Focus Areas

1. **CI/CD Pipeline Components**
   - Understand CodePipeline, CodeBuild, CodeDeploy
   - Know deployment strategies
   - Understand rollback mechanisms

2. **Infrastructure as Code**
   - CloudFormation template structure
   - Stack operations
   - Change sets and drift detection

3. **Monitoring and Logging**
   - CloudWatch metrics and alarms
   - Log aggregation
   - X-Ray for tracing

4. **Security Best Practices**
   - IAM roles vs users
   - Least privilege principle
   - Secrets management

5. **Container Services**
   - ECS vs EKS
   - Fargate for serverless containers
   - ECR for container images

### Common Scenarios

**Scenario 1**: Automated deployment with rollback
- Use CodePipeline with CodeDeploy
- Configure deployment configuration
- Enable automatic rollback on failures

**Scenario 2**: Multi-environment infrastructure
- Use CloudFormation with parameters
- Separate stacks per environment
- Use cross-stack references

**Scenario 3**: Secure credential management
- Use AWS Secrets Manager or Parameter Store
- IAM roles for applications
- Rotate credentials automatically

**Scenario 4**: Monitoring and alerting
- CloudWatch alarms for critical metrics
- SNS for notifications
- EventBridge for automation

---

## Practice Labs

### Lab 1: Create CI/CD Pipeline
1. Create CodeCommit repository
2. Set up CodeBuild project
3. Configure CodeDeploy application
4. Create CodePipeline connecting all stages

### Lab 2: Infrastructure as Code
1. Write CloudFormation template for VPC
2. Create EC2 instances in multiple AZs
3. Add Application Load Balancer
4. Implement Auto Scaling

### Lab 3: Monitoring Setup
1. Create CloudWatch dashboard
2. Set up custom metrics
3. Configure alarms
4. Create SNS notifications

---

## Additional Resources

- AWS DevOps Documentation: https://aws.amazon.com/devops/
- AWS Well-Architected Framework
- AWS DevOps Blog
- AWS Training and Certification

---

## Summary

AWS DevOps combines powerful AWS services with proven DevOps practices to enable:
- Faster development cycles
- More reliable deployments
- Better collaboration
- Improved security
- Scalable infrastructure

Master these concepts and practices to successfully implement DevOps on AWS and prepare for certification exams.

---

**Source**: DevOps AWS Training Materials (Korean)  
**Generated**: December 5, 2025  
**Format**: Study Guide for AWS Certification Preparation
