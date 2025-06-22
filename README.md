# Ooredoo Upsell Revenue Predictor V2

🚀 **Advanced AI-Powered Customer Upselling with Revenue Prediction & Strategic Planning**

_Enhanced version with regression modeling and comprehensive Azure OpenAI integration_

## 🌟 Overview

Ooredoo Upsell Revenue Predictor V2 is an advanced machine learning application that predicts potential monthly revenue increase from customer upselling opportunities. Unlike traditional classification models, this version uses **regression analysis** to provide precise revenue predictions and comprehensive strategic planning powered by Azure OpenAI.

## ✨ Key Features

### 🎯 Core Functionality

- **Revenue Regression Prediction**: Predicts exact QAR amount of potential monthly revenue increase
- **Advanced Feature Engineering**: 22+ engineered features including usage ratios and behavioral patterns
- **Azure OpenAI Integration**: GPT-4 powered customer analysis and strategic planning
- **Real-time Predictions**: Instant revenue potential assessment with confidence scoring
- **Interactive Web Interface**: Modern, responsive UI with enhanced user experience

### 📊 Enhanced Analytics

- **Precise Revenue Forecasting**: Quantified revenue predictions in QAR
- **Customer Segmentation**: High/Medium/Low value customer identification
- **Usage Pattern Analysis**: Comprehensive behavioral insights
- **Feature Importance**: Understanding key drivers of upsell potential
- **Confidence Scoring**: AI-driven prediction reliability assessment

### 🤖 AI-Powered Insights

- **Customer Analysis**: GPT-4 generated behavioral and value assessments
- **Upsell Recommendations**: Specific service and plan suggestions
- **Strategic Planning**: Detailed action plans with timing and channels
- **Risk Assessment**: Churn prevention and retention strategies
- **Market Adaptation**: Country-specific recommendations

### 🏗️ Technical Architecture

- **Machine Learning**: Random Forest Regressor with advanced hyperparameters
- **Feature Engineering**: Automated ratio calculations and behavioral metrics
- **Data Processing**: Enhanced customer profiling with 18 core attributes
- **API Integration**: Azure OpenAI with fallback mechanisms
- **Database Support**: SQLite (local) and Azure SQL Database (production)
- **Deployment**: Azure App Service with automated CI/CD

## 🛠️ Technology Stack

- **Backend**: Python Flask with Gunicorn
- **Machine Learning**: scikit-learn (Random Forest Regressor)
- **AI Integration**: Azure OpenAI GPT-4
- **Data Processing**: pandas, numpy
- **Database**: SQLite / Azure SQL Database
- **Frontend**: Bootstrap 5.3, Font Awesome 6.4
- **Deployment**: Azure App Service, Azure CLI
- **Monitoring**: Application Insights ready

## 📋 Prerequisites

- Python 3.11+
- Azure CLI (for deployment)
- Azure subscription with OpenAI service
- Git

## 🚀 Quick Start

### Local Development

1. **Clone the repository**

   ```bash
   cd ooredoo-upsell-v2
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

5. **Access the application**
   - Main Interface: http://localhost:5000
   - Customer Database: http://localhost:5000/customers
   - Health Check: http://localhost:5000/health

### Azure Deployment

1. **Make deployment script executable**

   ```bash
   chmod +x deploy-azure.sh
   ```

2. **Run deployment script**

   ```bash
   ./deploy-azure.sh
   ```

3. **Follow the prompts** for Azure login and configuration

## 🔧 Configuration

### Environment Variables

| Variable             | Description           | Default              |
| -------------------- | --------------------- | -------------------- |
| `AZURE_SQL_SERVER`   | Azure SQL Server name | None                 |
| `AZURE_SQL_DATABASE` | Database name         | ooredoo-upsell-v2-db |
| `AZURE_SQL_USERNAME` | SQL username          | None                 |
| `AZURE_SQL_PASSWORD` | SQL password          | None                 |
| `FLASK_ENV`          | Flask environment     | production           |
| `PORT`               | Application port      | 5000                 |

### Azure OpenAI Configuration

The application uses the same Azure OpenAI credentials as the original app:

- **Endpoint**: https://upselloreedoo.openai.azure.com/
- **Model**: GPT-4
- **API Version**: 2025-01-01-preview

## 📊 Model Details

### Regression Model Features

**Core Features (18):**

- Customer demographics (age, gender, country, location)
- Usage patterns (data, calls, SMS, roaming, international)
- Service utilization (premium services, plan details)
- Customer behavior (tenure, complaints, satisfaction, payment method)

**Engineered Features (4):**

- `spend_per_gb`: Monthly spend efficiency
- `calls_per_month`: Average daily call usage
- `avg_monthly_complaints`: Complaint frequency
- `premium_ratio`: Premium service adoption rate
- `roaming_ratio`: International usage indicator

### Model Performance

- **Algorithm**: Random Forest Regressor
- **Estimators**: 200 trees
- **Max Depth**: 15
- **Features**: 22 total features
- **Target**: Monthly revenue increase potential (QAR)

### Prediction Output

```json
{
  "predicted_revenue_increase": 156.75,
  "confidence_score": 0.82,
  "feature_importance": {
    "monthly_spend": 0.15,
    "satisfaction_score": 0.12,
    "data_usage_gb": 0.11,
    "...": "..."
  }
}
```

## 🎯 API Endpoints

### Core Endpoints

- `GET /` - Main prediction interface
- `POST /predict` - Revenue prediction API
- `GET /customers` - Customer database viewer
- `GET /health` - Application health check
- `GET /api/model-status` - Model training status

### Prediction API

**Request:**

```json
{
  "age": 35,
  "gender": "Male",
  "country": "Qatar",
  "location": "Doha",
  "monthly_spend": 200,
  "data_usage_gb": 15,
  "call_minutes": 250,
  "sms_count": 120,
  "current_plan": "Shahry+ Select",
  "satisfaction_score": 8.5
}
```

**Response:**

```json
{
  "prediction": {
    "predicted_revenue_increase": 156.75,
    "confidence_score": 0.82
  },
  "gpt_analysis": {
    "analysis": "Customer shows strong upsell potential...",
    "recommendations": [
      "Consider premium data plans",
      "Explore international calling packages"
    ],
    "upsell_plan": "Implement targeted campaign..."
  }
}
```

## 🔍 Customer Analysis Features

### AI-Powered Insights

1. **Customer Analysis**: Behavioral pattern assessment
2. **Upsell Recommendations**: Specific service suggestions
3. **Strategic Planning**: Detailed implementation roadmap
4. **Risk Assessment**: Churn prevention strategies
5. **Market Adaptation**: Country-specific approaches

### Revenue Segmentation

- **High Value** (>QAR 100): Premium upsell candidates
- **Medium Value** (QAR 50-100): Standard upgrade opportunities
- **Low Value** (<QAR 50): Retention-focused strategies

## 🌍 Multi-Country Support

Supports all 9 Ooredoo markets with localized features:

- **Qatar**: Premium market with high ARPU potential
- **Kuwait**: Strong purchasing power, business focus
- **Oman**: Balanced consumer and enterprise
- **Algeria**: Price-sensitive, volume-based
- **Tunisia**: Digital services adoption
- **Iraq**: Emerging market opportunities
- **Palestine**: Community-focused services
- **Maldives**: Tourism and connectivity
- **Myanmar**: Growth market potential

## 📈 Performance Monitoring

### Health Monitoring

```bash
# Check application health
curl https://your-app.azurewebsites.net/health
```

### Logging

```bash
# View live logs
az webapp log tail --name ooredoo-upsell-v2 --resource-group ooredoo-rg
```

### Metrics

- Model prediction accuracy
- Response times
- Error rates
- Customer segmentation distribution

## 🔒 Security Features

- **HTTPS Enforcement**: Automatic redirect to secure connections
- **Security Headers**: XSS protection, content type validation
- **Input Validation**: Comprehensive request sanitization
- **Error Handling**: Secure error responses
- **Environment Variables**: Secure credential management

## 🚀 Deployment Architecture

### Azure Resources

- **App Service**: Linux-based Python 3.11 runtime
- **App Service Plan**: B1 Basic tier (scalable)
- **Application Insights**: Performance monitoring
- **Azure SQL Database**: Production data storage
- **Azure OpenAI**: GPT-4 integration

### CI/CD Pipeline

1. **Code Deployment**: Azure CLI automated deployment
2. **Dependency Installation**: Automated pip install
3. **Health Checks**: Post-deployment validation
4. **Logging Setup**: Comprehensive monitoring
5. **Performance Testing**: Load and response validation

## 🔧 Troubleshooting

### Common Issues

1. **Model Training Fails**

   - Check data generation process
   - Verify feature engineering pipeline
   - Review memory allocation

2. **Azure OpenAI Errors**

   - Validate API credentials
   - Check rate limits
   - Review endpoint configuration

3. **Database Connection Issues**
   - Verify Azure SQL credentials
   - Check firewall settings
   - Test SQLite fallback

### Debug Commands

```bash
# Check model status
curl https://your-app.azurewebsites.net/api/model-status

# View application logs
az webapp log download --name ooredoo-upsell-v2 --resource-group ooredoo-rg

# Restart application
az webapp restart --name ooredoo-upsell-v2 --resource-group ooredoo-rg
```

## 📚 Documentation

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure OpenAI Service](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is proprietary to Ooredoo Group.

## 📞 Support

For technical support or questions:

- Create an issue in the repository
- Contact the development team
- Review the troubleshooting guide

---

**Ooredoo Upsell Revenue Predictor V2** - Transforming customer insights into revenue growth through AI-powered predictions and strategic planning.
