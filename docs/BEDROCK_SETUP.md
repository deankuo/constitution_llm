# AWS Bedrock Configuration Guide

This guide explains how to configure AWS Bedrock models for the Historical Political Indicators LLM project.

## Prerequisites

1. **AWS Account** with Bedrock access
2. **AWS Credentials** (Access Key ID and Secret Access Key)
3. **Bedrock Model Access** - Request access to Claude models in AWS Bedrock console

## Finding Your Bedrock Model ARN

### Option 1: Using AWS Console

1. Go to AWS Bedrock Console → Model Access
2. Find your enabled model (e.g., Claude Sonnet 4.5)
3. The ARN format is: `arn:aws:bedrock:region:account-id:inference-profile/model-id`

### Option 2: Using AWS CLI

```bash
# List available models
aws bedrock list-foundation-models --region us-east-1

# Get inference profile ARN
aws bedrock get-inference-profile --inference-profile-id global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

## Configuration Methods

### Method 1: Environment Variables (Recommended for Public Repos)

Add to your `.env` file:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_SESSION_TOKEN=your_session_token  # Optional, for temporary credentials
AWS_REGION=us-east-1

# Bedrock Model Configuration
# Option A: Use full ARN (if you have a specific inference profile)
BEDROCK_VERIFIER_MODEL=arn:aws:bedrock:us-east-1:979986032704:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0

# Option B: Use model ID only (works with your AWS credentials)
BEDROCK_VERIFIER_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Method 2: Command Line Arguments

```bash
python main.py --pipeline leader \
    --verifier-model "arn:aws:bedrock:us-east-1:ACCOUNT_ID:inference-profile/MODEL_ID" \
    --verify cove \
    --verify-indicators constitution
```

### Method 3: Direct Code Configuration

In `config.py`, update the default:

```python
DEFAULT_VERIFIER_MODEL = "your-full-arn-or-model-id"
```

**⚠️ Warning**: Do NOT commit account-specific ARNs to public repositories!

## Supported Bedrock Model Formats

The system automatically detects Bedrock models by these patterns:

1. **Full ARN**: `arn:aws:bedrock:*`
2. **Inference Profile ARN**: `arn:aws:bedrock:region:account:inference-profile/model-id`
3. **Model ID with provider prefix**: `anthropic.*`, `amazon.*`, `meta.*`, `cohere.*`, etc.

### Example Model IDs

```bash
# Anthropic Claude models
anthropic.claude-sonnet-4-5-20250929-v1:0
anthropic.claude-3-5-sonnet-20241022-v2:0
anthropic.claude-3-5-haiku-20241022-v1:0

# Amazon Titan models
amazon.titan-text-premier-v1:0

# Meta Llama models
meta.llama3-2-90b-instruct-v1:0
```

## Cost Tracking

The system automatically extracts the model identifier from ARNs for cost tracking:

```
ARN: arn:aws:bedrock:us-east-1:123456:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0
Extracted for pricing: global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

Cost logs are saved to `data/logs/{experiment}_costs.json` with per-model breakdowns.

## Testing Your Configuration

Run a test with 5 samples:

```bash
python main.py --pipeline leader \
    --mode multiple \
    --indicators constitution \
    --verify cove \
    --verify-indicators constitution \
    --test 5 \
    --input data/plt_leaders_data.csv \
    --output data/results/bedrock_test.csv
```

Check the cost log at `data/logs/bedrock_test_costs.json` to verify model usage.

## Troubleshooting

### Error: "Anthropic API key not provided"

**Cause**: System detected model as Anthropic direct API instead of Bedrock

**Solution**: Ensure your model identifier includes provider prefix:
- ✅ `anthropic.claude-sonnet-4-5-20250929-v1:0`
- ❌ `claude-sonnet-4-5-20250929-v1:0`

### Error: "Could not connect to endpoint"

**Cause**: AWS credentials not configured or region mismatch

**Solution**:
1. Verify AWS credentials in `.env`
2. Check `AWS_REGION` matches your Bedrock model region
3. Ensure Bedrock service is available in your region

### Error: "ResourceNotFoundException"

**Cause**: Model not available in your account/region

**Solution**:
1. Go to Bedrock Console → Model Access
2. Request access to the model
3. Wait for approval (usually instant for Claude models)

## Best Practices for Public Repositories

1. **Use `.env` for credentials** - Never commit `.env` to git
2. **Provide `.env.example`** with placeholder values
3. **Document model ID format** - Use model IDs without account-specific ARNs in documentation
4. **Test with environment variables** - Ensure code works without hardcoded ARNs

Example `.env.example`:

```env
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# Bedrock Model (use model ID or your full ARN)
BEDROCK_VERIFIER_MODEL=anthropic.claude-sonnet-4-5-20250929-v1:0
```

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Claude Models on Bedrock](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock)
