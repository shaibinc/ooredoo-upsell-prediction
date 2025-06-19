# GPT API Integration Setup Guide

This guide explains how to set up and configure the GPT API integration for enhanced upsell predictions with natural language explanations and marketing recommendations.

## 🚀 Quick Start

The GPT API integration is currently **disabled by default** and uses intelligent fallback explanations. To enable GPT-powered insights:

1. **Set up Azure OpenAI or OpenAI API**
2. **Configure API credentials**
3. **Enable the integration**

## 🔧 Configuration Options

### Option 1: Azure OpenAI (Recommended for Enterprise)

1. **Provision Azure OpenAI Service**:
   - Go to Azure Portal → Create Resource → Azure OpenAI
   - Deploy a GPT model (GPT-3.5 Turbo or GPT-4)
   - Note your endpoint URL and API key

2. **Update Configuration in `app.py`**:
   ```python
   # GPT API Configuration
   GPT_API_ENABLED = True  # Enable the integration
   OPENAI_API_TYPE = "azure"
   OPENAI_API_BASE = "https://your-azure-openai-endpoint.openai.azure.com/"
   OPENAI_API_VERSION = "2025-01-01-preview"
   OPENAI_API_KEY = "your-azure-openai-api-key"
   OPENAI_ENGINE = "gpt-4"  # e.g., "gpt-35-turbo"
   ```

### Option 2: Direct OpenAI API

1. **Get OpenAI API Key**:
   - Visit https://platform.openai.com/api-keys
   - Create a new API key

2. **Update Configuration in `app.py`**:
   ```python
   # GPT API Configuration
   GPT_API_ENABLED = True
   OPENAI_API_TYPE = "openai"
   OPENAI_API_BASE = "https://api.openai.com/v1"
   OPENAI_API_KEY = "your-openai-api-key"
   OPENAI_ENGINE = "gpt-3.5-turbo"  # or "gpt-4"
   ```

## 🔒 Security Best Practices

### Environment Variables (Recommended)

Instead of hardcoding API keys, use environment variables:

```python
import os

# GPT API Configuration
GPT_API_ENABLED = os.getenv('GPT_API_ENABLED', 'False').lower() == 'true'
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE', 'azure')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION', '2023-05-15')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENGINE = os.getenv('OPENAI_ENGINE', 'gpt-35-turbo')
```

Then set environment variables:
```bash
export GPT_API_ENABLED=true
export OPENAI_API_KEY=your-api-key
export OPENAI_API_BASE=your-endpoint
export OPENAI_ENGINE=your-model-name
```

### Azure Key Vault (Enterprise)

For production deployments, store secrets in Azure Key Vault:

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Initialize Key Vault client
credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)

# Retrieve secrets
OPENAI_API_KEY = client.get_secret("openai-api-key").value
```

## 📊 Features Enabled by GPT Integration

### 1. **Natural Language Explanations**
- Plain English explanations of why the model predicts high/medium/low upsell probability
- Based on customer behavior patterns, usage data, and demographics
- Helps non-technical users understand AI predictions

### 2. **Personalized Marketing Recommendations**
- Specific, actionable marketing suggestions for each customer
- Tailored to customer profile and predicted upsell likelihood
- Ready-to-use recommendations for marketing teams

### 3. **Intelligent Fallbacks**
- When GPT API is disabled or fails, the system provides intelligent fallback explanations
- Ensures the application continues to work seamlessly
- No degradation in core prediction functionality

## 🎯 Sample GPT Outputs

### High Priority Customer
**Explanation**: "Customer frequently exceeds data limits with 25GB monthly usage and shows consistent high spending patterns, indicating strong appetite for premium services."

**Marketing Recommendation**: "Offer unlimited data plan with 5G priority access at 20% discount for first 3 months to capitalize on high usage patterns."

### Medium Priority Customer
**Explanation**: "Moderate data usage and tenure suggest potential for upgrade, but complaint history indicates need for value demonstration."

**Marketing Recommendation**: "Present mid-tier plan upgrade with clear benefits comparison and customer service priority to address past concerns."

### Low Priority Customer
**Explanation**: "Low monthly spend and minimal data usage indicate price sensitivity and basic service needs."

**Marketing Recommendation**: "Focus on retention with small value-adds like free SMS bundles or loyalty rewards rather than plan upgrades."

## 💰 Cost Management

### API Usage Optimization
- **Prompt Length**: Optimized prompts (≤150 tokens) to minimize costs
- **Response Limits**: Max 150 tokens per response
- **Timeout**: 10-second timeout to prevent hanging requests
- **Fallback Strategy**: Immediate fallback on API errors

### Estimated Costs
- **GPT-3.5 Turbo**: ~$0.002 per prediction
- **GPT-4**: ~$0.06 per prediction
- **Monthly for 1000 predictions**: $2-60 depending on model

## 🔧 Installation & Testing

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Configuration**:
   ```python
   # Test GPT API connection
   from app import get_gpt_explanation
   
   test_customer = {
       'age': 30, 'gender': 'Male', 'monthly_spend': 75,
       'data_usage_gb': 20, 'tenure_months': 12
   }
   
   result = get_gpt_explanation(test_customer, 0.8, 'High')
   print(result)
   ```

3. **Monitor Logs**:
   - Check console for "GPT API Error" messages
   - Verify fallback explanations are working

## 🚨 Troubleshooting

### Common Issues

1. **"GPT API Error: Invalid API key"**
   - Verify API key is correct
   - Check if key has proper permissions

2. **"GPT API Error: Model not found"**
   - Ensure model is deployed in Azure OpenAI
   - Verify engine name matches deployment

3. **"GPT API Error: Timeout"**
   - Network connectivity issues
   - API service temporarily unavailable
   - Fallback explanations will be used

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Monitoring & Analytics

### Track GPT Usage
- Monitor API call frequency
- Track response times
- Measure explanation quality
- Monitor fallback usage rates

### Performance Metrics
- **API Success Rate**: Target >95%
- **Response Time**: Target <2 seconds
- **Cost per Prediction**: Monitor monthly spend
- **User Satisfaction**: Track explanation helpfulness

## 🔄 Future Enhancements

1. **Multi-language Support**: Explanations in Arabic, English
2. **Custom Prompts**: Industry-specific explanation templates
3. **A/B Testing**: Compare GPT vs. rule-based explanations
4. **Feedback Loop**: Learn from user feedback to improve prompts
5. **Batch Processing**: Optimize for bulk predictions

---

**Note**: The application works perfectly without GPT integration using intelligent fallback explanations. GPT integration is an enhancement that provides more personalized and natural language insights.