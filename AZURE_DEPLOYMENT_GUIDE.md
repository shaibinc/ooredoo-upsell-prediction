# Azure Deployment Guide for Ooredoo Upsell Prediction App

This guide provides step-by-step instructions to deploy the Ooredoo Customer Upsell Prediction web application to Azure App Service.

## Prerequisites

1. **Azure Account**: Active Azure subscription
2. **Azure CLI**: Install Azure CLI on your local machine
3. **Git**: For code deployment
4. **Python 3.9+**: For local testing

## Deployment Methods

### Method 1: Azure Portal Deployment (Recommended for beginners)

#### Step 1: Create Azure App Service

1. Log in to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → "Web App"
3. Configure the following:
   - **Subscription**: Select your subscription
   - **Resource Group**: Create new or select existing
   - **Name**: `ooredoo-upsell-app` (must be globally unique)
   - **Runtime Stack**: Python 3.9
   - **Operating System**: Linux
   - **Region**: Choose closest to your users
   - **Pricing Plan**: B1 Basic (recommended for production)

#### Step 2: Configure Deployment

1. In your App Service, go to "Deployment Center"
2. Choose deployment source:
   - **GitHub**: Connect your GitHub repository
   - **Local Git**: Use Azure's Git repository
   - **ZIP Deploy**: Upload code directly

#### Step 3: Configure Application Settings

1. Go to "Configuration" → "Application Settings"
2. Add the following settings:

   ```
   FLASK_ENV=production
   PYTHONPATH=/home/site/wwwroot
   SCM_DO_BUILD_DURING_DEPLOYMENT=true
   ```

3. **For GPT API Integration** (if enabled):
   ```
   OPENAI_API_TYPE=azure
   OPENAI_API_BASE=https://your-resource.openai.azure.com/
   OPENAI_API_VERSION=2023-05-15
   OPENAI_API_KEY=your-api-key
   OPENAI_ENGINE=your-deployment-name
   GPT_API_ENABLED=true
   ```

#### Step 4: Configure Startup Command

1. In "Configuration" → "General Settings"
2. Set **Startup Command**: `bash startup.sh`

### Method 2: Azure CLI Deployment

#### Step 1: Login and Setup

```bash
# Login to Azure
az login

# Create resource group
az group create --name ooredoo-rg --location "East US"

# Create App Service Plan
az appservice plan create --name ooredoo-plan --resource-group ooredoo-rg --sku B1 --is-linux
```

#### Step 2: Create Web App

```bash
# Create web app
az webapp create --resource-group ooredoo-rg --plan ooredoo-plan --name ooredoo-upsell-app --runtime "PYTHON|3.9"

# Configure startup command
az webapp config set --resource-group ooredoo-rg --name ooredoo-upsell-app --startup-file "bash startup.sh"
```

#### Step 3: Deploy Code

```bash
# Deploy from local Git
az webapp deployment source config-local-git --name ooredoo-upsell-app --resource-group ooredoo-rg

# Add Azure remote and push
git remote add azure <deployment-url>
git push azure main
```

### Method 3: GitHub Actions (CI/CD)

1. Fork/clone the repository to your GitHub account
2. In Azure Portal, go to your App Service → "Deployment Center"
3. Select "GitHub" and authorize
4. Select your repository and branch
5. Azure will automatically create a GitHub Actions workflow

## Post-Deployment Configuration

### 1. Verify Deployment

1. Navigate to your app URL: `https://ooredoo-upsell-app.azurewebsites.net`
2. Check that the application loads correctly
3. Test prediction functionality

### 2. Monitor Application

1. **Application Insights**: Enable for monitoring and diagnostics
2. **Log Stream**: View real-time logs in Azure Portal
3. **Metrics**: Monitor CPU, memory, and request metrics

### 3. Configure Custom Domain (Optional)

1. Go to "Custom domains" in your App Service
2. Add your custom domain
3. Configure SSL certificate

## Environment Variables for Production

### Required Settings

```
FLASK_ENV=production
PYTHONPATH=/home/site/wwwroot
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

### GPT API Settings (if using)

```
OPENAI_API_TYPE=azure
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2023-05-15
OPENAI_API_KEY=your-api-key
OPENAI_ENGINE=your-deployment-name
GPT_API_ENABLED=true
```

## Security Best Practices

1. **API Keys**: Store in Azure Key Vault or App Service Configuration
2. **HTTPS**: Enable HTTPS-only in App Service settings
3. **Authentication**: Configure Azure AD if needed
4. **Network**: Use VNet integration for enhanced security

## Scaling and Performance

### Auto-scaling

1. Go to "Scale out (App Service plan)"
2. Enable auto-scaling based on:
   - CPU percentage
   - Memory percentage
   - Request count

### Performance Optimization

1. **Application Insights**: Monitor performance bottlenecks
2. **CDN**: Use Azure CDN for static assets
3. **Caching**: Implement Redis cache if needed

## Troubleshooting

### Common Issues

1. **Application won't start**:

   - Check startup command in Configuration
   - Verify Python version compatibility
   - Check application logs

2. **Database issues**:

   - Ensure SQLite file permissions
   - Consider migrating to Azure SQL Database for production

3. **GPT API errors**:
   - Verify API keys and endpoints
   - Check quota and rate limits
   - Review error logs

### Debugging

1. **Log Stream**: Real-time application logs
2. **Kudu Console**: Access file system and run commands
3. **Application Insights**: Detailed telemetry and error tracking

## Cost Optimization

1. **App Service Plan**: Choose appropriate tier
2. **Auto-scaling**: Scale down during low usage
3. **Reserved Instances**: For predictable workloads
4. **Monitoring**: Set up cost alerts

## Backup and Disaster Recovery

1. **App Service Backup**: Configure automatic backups
2. **Database Backup**: Regular SQLite file backups
3. **Source Control**: Maintain code in Git repository
4. **Documentation**: Keep deployment procedures updated

## Support and Maintenance

1. **Updates**: Regular dependency updates
2. **Security**: Monitor security advisories
3. **Performance**: Regular performance reviews
4. **Monitoring**: Set up alerts for critical metrics

---

## Quick Deployment Checklist

- [ ] Azure subscription active
- [ ] Resource group created
- [ ] App Service created with Python 3.9
- [ ] Startup command configured
- [ ] Environment variables set
- [ ] Code deployed
- [ ] Application tested
- [ ] Monitoring enabled
- [ ] Security configured
- [ ] Documentation updated

For additional support, refer to [Azure App Service documentation](https://docs.microsoft.com/en-us/azure/app-service/) or contact your Azure support team.
