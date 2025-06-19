# Azure SQL Database Setup for Ooredoo Upsell Prediction App

This guide explains how to set up Azure SQL Database for the Ooredoo Upsell Prediction App.

## Prerequisites

1. Azure subscription
2. Azure SQL Database instance
3. Azure App Service configured

## Step 1: Create Azure SQL Database

1. Go to Azure Portal
2. Create a new SQL Database:
   - Resource group: Choose or create one
   - Database name: `ooredoo-db`
   - Server: Create new or use existing
   - Pricing tier: Choose based on your needs (Basic/Standard recommended for development)

## Step 2: Configure Firewall Rules

1. In your SQL Server settings, go to "Firewalls and virtual networks"
2. Add your Azure App Service IP ranges
3. Enable "Allow Azure services and resources to access this server"

## Step 3: Set Environment Variables in Azure App Service

In your Azure App Service, go to Configuration > Application settings and add:

```
AZURE_SQL_SERVER=your-server-name.database.windows.net
AZURE_SQL_DATABASE=ooredoo-db
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password
```

## Step 4: Database Schema

The app will automatically create the required table structure:

```sql
CREATE TABLE customers (
    customer_id INT IDENTITY(1,1) PRIMARY KEY,
    age INT,
    gender NVARCHAR(50),
    country NVARCHAR(100),
    location NVARCHAR(100),
    monthly_spend DECIMAL(10,2),
    data_usage_gb DECIMAL(10,2),
    call_minutes INT,
    sms_count INT,
    tenure_months INT,
    complaint_count INT,
    payment_method NVARCHAR(50),
    current_plan NVARCHAR(100),
    upsell_target DECIMAL(10,2),
    created_at DATETIME2 DEFAULT GETDATE()
)
```

## Step 5: Deploy and Test

1. Deploy your app to Azure App Service
2. Check the logs to ensure database connection is successful
3. The app will automatically populate sample data on first run

## Troubleshooting

### Connection Issues
- Verify firewall rules allow Azure App Service
- Check environment variables are set correctly
- Ensure SQL Server allows Azure services

### Authentication Issues
- Verify username and password
- Check if SQL authentication is enabled
- Consider using Azure AD authentication for production

### Performance
- Monitor DTU usage in Azure portal
- Consider upgrading pricing tier if needed
- Implement connection pooling for high-traffic scenarios

## Security Best Practices

1. Use Azure Key Vault for storing connection strings
2. Enable Azure AD authentication
3. Use managed identities when possible
4. Regularly update passwords
5. Monitor access logs

## Cost Optimization

1. Use Basic tier for development/testing
2. Scale down during non-business hours
3. Monitor DTU usage and optimize queries
4. Consider serverless options for variable workloads