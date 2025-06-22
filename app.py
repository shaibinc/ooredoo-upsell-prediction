import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import sys
import traceback
from datetime import datetime
import sqlite3
import json
import warnings
import openai
import requests
from flask import Flask, render_template, request, jsonify
import time
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Database configuration
import tempfile
import os

# Database configuration
if os.environ.get('WEBSITE_SITE_NAME'):  # Running in Azure
    # Check if Azure SQL credentials are available
    azure_sql_server = os.environ.get('AZURE_SQL_SERVER')
    azure_sql_username = os.environ.get('AZURE_SQL_USERNAME')
    azure_sql_password = os.environ.get('AZURE_SQL_PASSWORD')
    
    if azure_sql_server and azure_sql_username and azure_sql_password:
        # Use Azure SQL Database
        USE_AZURE_SQL = True
        AZURE_SQL_SERVER = azure_sql_server
        AZURE_SQL_DATABASE = os.environ.get('AZURE_SQL_DATABASE', 'ooredoo-upsell-v2-db')
        AZURE_SQL_USERNAME = azure_sql_username
        AZURE_SQL_PASSWORD = azure_sql_password
        DATABASE = None  # Not used for Azure SQL
    else:
        # Fall back to SQLite if Azure SQL credentials are not available
        print("Azure SQL credentials not found, falling back to SQLite")
        USE_AZURE_SQL = False
        DATABASE = 'ooredoo_upsell_v2.db'
else:  # Running locally
    USE_AZURE_SQL = False
    DATABASE = 'ooredoo_upsell_v2.db'

# Import pyodbc for Azure SQL Database
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    print("Warning: pyodbc not available. Azure SQL Database features will be disabled.")

# Global database connection for in-memory database in Azure
_global_db_connection = None

def get_db_connection():
    """Get database connection for SQLite or Azure SQL Database"""
    global _global_db_connection
    if USE_AZURE_SQL and PYODBC_AVAILABLE:
        try:
            # Try Azure SQL Database connection
            if AZURE_SQL_USERNAME and AZURE_SQL_PASSWORD:
                # Use SQL Server authentication with username/password
                connection_string = (
                    f"Driver={{ODBC Driver 18 for SQL Server}};"
                    f"Server=tcp:{AZURE_SQL_SERVER},1433;"
                    f"Database={AZURE_SQL_DATABASE};"
                    f"Uid={AZURE_SQL_USERNAME};"
                    f"Pwd={AZURE_SQL_PASSWORD};"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                    f"Connection Timeout=30;"
                )
            else:
                # Use Managed Identity authentication
                connection_string = (
                    f"Driver={{ODBC Driver 18 for SQL Server}};"
                    f"Server=tcp:{AZURE_SQL_SERVER},1433;"
                    f"Database={AZURE_SQL_DATABASE};"
                    f"Authentication=ActiveDirectoryMsi;"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                    f"Connection Timeout=30;"
                )
            return pyodbc.connect(connection_string)
        except Exception as e:
            print(f"Failed to connect to Azure SQL Database: {e}")
            print("Falling back to SQLite database")
            # Fall back to SQLite
            return sqlite3.connect('ooredoo_upsell_v2.db')
    elif DATABASE == ":memory:" and _global_db_connection:
        return _global_db_connection
    else:
        # SQLite connection for local development
        return sqlite3.connect(DATABASE if DATABASE else 'ooredoo_upsell_v2.db')

# OpenAI Configuration - Using regular OpenAI API
GPT_API_ENABLED = True
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = "gpt-4"

# Configure OpenAI if enabled
if GPT_API_ENABLED:
    openai.api_key = OPENAI_API_KEY

class OoreedooUpsellRegressionPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def generate_sample_data(self, n_samples=200):
        """Generate enhanced sample customer data for Ooredoo Qatar with regression targets"""
        # Use random seed based on current time for fresh data each initialization
        np.random.seed(int(time.time()) % 10000)
        
        # Customer demographics
        customer_ids = list(range(1, n_samples + 1))
        
        # Generate more realistic customer profiles with wider range including poor customers
        ages = np.random.choice([18, 25, 35, 45, 55, 65, 75], n_samples, p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.08, 0.02])
        genders = np.random.choice(['Male', 'Female'], n_samples)
        
        # Focus on Qatar customers only
        countries = np.full(n_samples, 'Qatar')
        
        # Country-specific locations mapping
        country_locations = {
            'Qatar': ['Doha', 'Al Rayyan', 'Al Wakrah', 'Al Khor', 'Umm Salal'],
            'Oman': ['Muscat', 'Salalah', 'Nizwa', 'Sur', 'Sohar'],
            'Kuwait': ['Kuwait City', 'Hawalli', 'Farwaniya', 'Ahmadi', 'Jahra'],
            'Algeria': ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'],
            'Tunisia': ['Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte'],
            'Iraq': ['Baghdad', 'Basra', 'Mosul', 'Erbil', 'Najaf'],
            'Palestine': ['Gaza', 'Ramallah', 'Hebron', 'Nablus', 'Bethlehem'],
            'Maldives': ['Male', 'Addu City', 'Fuvahmulah', 'Kulhudhuffushi', 'Thinadhoo'],
            'Myanmar': ['Yangon', 'Mandalay', 'Naypyidaw', 'Mawlamyine', 'Bago']
        }
        
        # Generate locations for Qatar customers only
        qatar_locations = ['Doha', 'Al Rayyan', 'Al Wakrah', 'Al Khor', 'Umm Salal']
        locations = np.random.choice(qatar_locations, n_samples)
        
        # Create more diverse spending patterns including very low spenders
        # 20% of customers are low spenders (15-40 QAR)
        low_spenders = int(n_samples * 0.2)
        normal_spenders = n_samples - low_spenders
        
        low_monthly_spends = np.random.uniform(15, 40, low_spenders)
        normal_monthly_spends = np.random.exponential(250, normal_spenders) + 75
        monthly_spend = np.concatenate([low_monthly_spends, normal_monthly_spends])
        np.random.shuffle(monthly_spend)
        
        # Data usage - low spenders use very little data
        data_usage_gb = np.where(monthly_spend < 50, 
                                np.random.uniform(0.1, 2, n_samples),  # Low usage for low spenders
                                np.random.exponential(20, n_samples) + 8)   # Normal usage for others
        data_usage_gb = np.clip(data_usage_gb, 0.1, 100)
        
        # Call minutes - low spenders make fewer calls
        call_minutes = np.where(monthly_spend < 50,
                               np.random.uniform(10, 100, n_samples),  # Low calls for low spenders
                               np.random.exponential(300, n_samples) + 100)     # Normal calls for others
        call_minutes = np.clip(call_minutes, 10, 2000)
        
        sms_count = np.random.exponential(150, n_samples) + 20
        
        # Low spenders rarely use roaming or international
        roaming_usage = np.where(monthly_spend < 50, 0, np.random.exponential(50, n_samples))
        roaming_usage = np.clip(roaming_usage, 0, 200)
        
        international_calls = np.where(monthly_spend < 50, 0, np.random.exponential(30, n_samples))
        international_calls = np.clip(international_calls, 0, 300)
        
        premium_services = np.where(monthly_spend < 50, 0, np.random.exponential(25, n_samples))
        premium_services = np.clip(premium_services, 0, 50)
        
        # Customer behavior
        tenure_months = np.random.exponential(30, n_samples) + 3
        
        # Create more realistic complaint patterns - low spenders often have more complaints
        # Low spenders have higher complaint rates
        complaint_probs_low = [0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005]
        complaint_probs_normal = [0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002]
        
        # Normalize probabilities to ensure they sum to 1
        complaint_probs_low = np.array(complaint_probs_low)
        complaint_probs_low = complaint_probs_low / complaint_probs_low.sum()
        complaint_probs_normal = np.array(complaint_probs_normal)
        complaint_probs_normal = complaint_probs_normal / complaint_probs_normal.sum()
        
        complaint_count = np.where(monthly_spend < 50,
                                   np.random.choice(range(10), n_samples, p=complaint_probs_low),
                                   np.random.choice(range(10), n_samples, p=complaint_probs_normal))
        
        payment_method = np.random.choice(['Credit Card', 'Bank Transfer', 'Cash', 'Mobile Payment'], n_samples)
        
        # Create more realistic satisfaction scores - low spenders often less satisfied
        base_satisfaction = np.where(monthly_spend < 50,
                                    np.random.normal(5, 2, n_samples),  # Lower base satisfaction for low spenders
                                    np.random.normal(7.5, 1.5, n_samples))  # Higher for normal spenders
        # Reduce satisfaction based on complaints
        satisfaction_score = base_satisfaction - (complaint_count * 0.8)
        satisfaction_score = np.clip(satisfaction_score, 1, 10)
        
        # Current plan types with pricing tiers
        current_plans = np.random.choice([
            'Hala Super 40', 'Hala Digital', 'Hala Visitor SIM',
            'Shahry+ Select', 'Shahry+ Go', 'Qatarna+ Pro', 
            'Qatarna+ Premium', 'Qatarna+ Al Nokhba', 'Business Elite',
            'Enterprise Plus', 'Family Bundle', 'Student Special'
        ], n_samples)
        
        # Plan pricing (QAR per month)
        plan_pricing = {
            'Hala Super 40': 40, 'Hala Digital': 60, 'Hala Visitor SIM': 25,
            'Shahry+ Select': 80, 'Shahry+ Go': 120, 'Qatarna+ Pro': 150, 
            'Qatarna+ Premium': 200, 'Qatarna+ Al Nokhba': 300, 'Business Elite': 400,
            'Enterprise Plus': 500, 'Family Bundle': 180, 'Student Special': 35
        }
        
        current_plan_price = np.array([plan_pricing[plan] for plan in current_plans])
        
        # Country-specific economic factors
        country_factors = {
            'Qatar': 1.3, 'Kuwait': 1.2, 'Oman': 1.0, 'Algeria': 0.7, 
            'Tunisia': 0.7, 'Iraq': 0.6, 'Palestine': 0.5, 'Maldives': 0.9, 'Myanmar': 0.6
        }
        country_multipliers = np.array([country_factors[country] for country in countries])
        
        # Calculate regression target: Potential Monthly Revenue Increase (QAR)
        # This is what we want to predict - how much additional revenue we can get from upselling
        
        # Create a more realistic formula that can produce very low values for poor customers
        satisfaction_multiplier = np.maximum(satisfaction_score / 5.0, 0.2)  # Scale satisfaction impact
        complaint_penalty = complaint_count * 25  # Heavy penalty for complaints
        
        base_upsell_potential = (
            (monthly_spend * 0.2 * satisfaction_multiplier) +  # Spend scaled by satisfaction
            (data_usage_gb * 2) +     # Data usage factor
            (call_minutes * 0.05) +   # Call usage factor
            (roaming_usage * 0.8) +   # Roaming indicates higher value customer
            (international_calls * 1.0) +  # International usage
            (premium_services * 0.8) +  # Premium service usage
            (tenure_months * 1.0) -     # Loyalty bonus
            complaint_penalty           # Heavy complaint penalty
        )
        
        # Apply satisfaction multiplier to the whole calculation
        base_upsell_potential = base_upsell_potential * satisfaction_multiplier
        
        # Apply country economic factors
        upsell_revenue_potential = base_upsell_potential * country_multipliers
        
        # Add some noise and create realistic range including low values
        upsell_revenue_potential += np.random.normal(0, 15, n_samples)
        # Allow negative values for truly poor customers - they may cost more than they generate
        # No minimum cap - let the model learn the full range
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'country': countries,
            'location': locations,
            'monthly_spend': monthly_spend,
            'data_usage_gb': data_usage_gb,
            'call_minutes': call_minutes,
            'sms_count': sms_count,
            'roaming_usage': roaming_usage,
            'international_calls': international_calls,
            'premium_services': premium_services,
            'tenure_months': tenure_months,
            'complaint_count': complaint_count,
            'payment_method': payment_method,
            'satisfaction_score': satisfaction_score,
            'current_plan': current_plans,
            'current_plan_price': current_plan_price,
            'upsell_revenue_potential': upsell_revenue_potential
        })
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for regression model"""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Categorical columns to encode
        categorical_columns = ['gender', 'country', 'location', 'payment_method', 'current_plan']
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # Handle unseen categories
                unique_values = set(df_processed[col].unique())
                known_values = set(self.label_encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to the encoder
                    all_values = list(known_values) + list(new_values)
                    self.label_encoders[col].classes_ = np.array(all_values)
                
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Feature engineering
        df_processed['spend_per_gb'] = df_processed['monthly_spend'] / (df_processed['data_usage_gb'] + 1)
        df_processed['calls_per_month'] = df_processed['call_minutes'] / 30
        df_processed['avg_monthly_complaints'] = df_processed['complaint_count'] / (df_processed['tenure_months'] + 1)
        df_processed['premium_ratio'] = df_processed['premium_services'] / (df_processed['monthly_spend'] + 1)
        df_processed['roaming_ratio'] = df_processed['roaming_usage'] / (df_processed['monthly_spend'] + 1)
        
        # Select features for training
        feature_columns = [
            'age', 'gender', 'country', 'location', 'monthly_spend', 'data_usage_gb',
            'call_minutes', 'sms_count', 'roaming_usage', 'international_calls',
            'premium_services', 'tenure_months', 'complaint_count', 'payment_method',
            'satisfaction_score', 'current_plan', 'current_plan_price',
            'spend_per_gb', 'calls_per_month', 'avg_monthly_complaints',
            'premium_ratio', 'roaming_ratio'
        ]
        
        self.feature_columns = feature_columns
        return df_processed[feature_columns]
    
    def train_model(self, df):
        """Train the regression model"""
        print("Training regression model for upsell revenue prediction...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['upsell_revenue_potential']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        
        return {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict_upsell_potential(self, customer_data):
        """Predict upsell revenue potential for a customer"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare customer data
        df_customer = pd.DataFrame([customer_data])
        X_customer = self.prepare_features(df_customer)
        X_customer_scaled = self.scaler.transform(X_customer)
        
        # Make prediction
        predicted_revenue = self.model.predict(X_customer_scaled)[0]
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Calculate confidence score based on absolute value and model certainty
        # Higher absolute values (positive or negative) indicate higher confidence
        confidence_score = min(max(abs(predicted_revenue) / 200, 0.1), 1.0)  # Normalize to 0.1-1.0
        
        return {
            'predicted_revenue_increase': round(predicted_revenue, 2),
            'confidence_score': round(confidence_score, 2),
            'feature_importance': feature_importance
        }

# Global predictor instance
predictor = None
model_trained = False
training_in_progress = False

# Initialize database and model when module is imported
def initialize_app():
    """Initialize the application - called when module is imported"""
    global model_trained, training_in_progress
    if not model_trained and not training_in_progress:
        try:
            print("Auto-initializing database and model...")
            init_database()
        except Exception as e:
            print(f"Error during auto-initialization: {e}")
            traceback.print_exc()

def init_database():
    """Initialize the database"""
    print("Initializing database for Upsell Revenue Predictor...")
    
    if USE_AZURE_SQL:
        print("Azure SQL Database detected, initializing...")
        if not PYODBC_AVAILABLE:
            print("Error: pyodbc not available for Azure SQL Database")
            raise ImportError("pyodbc is required for Azure SQL Database")
    else:
        # Local SQLite environment
        if os.path.exists(DATABASE):
            print(f"Removing existing database: {DATABASE}")
            os.remove(DATABASE)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create customers table with additional fields for regression
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            country TEXT,
            location TEXT,
            monthly_spend REAL,
            data_usage_gb REAL,
            call_minutes REAL,
            sms_count REAL,
            roaming_usage REAL,
            international_calls REAL,
            premium_services REAL,
            tenure_months REAL,
            complaint_count INTEGER,
            payment_method TEXT,
            satisfaction_score REAL,
            current_plan TEXT,
            current_plan_price REAL,
            upsell_revenue_potential REAL
        )
        '''
        
        cursor.execute(create_table_sql)
        
        # Generate and insert sample data
        global predictor
        predictor = OoreedooUpsellRegressionPredictor()
        df = predictor.generate_sample_data(200)
        
        # Insert data into database
        for _, row in df.iterrows():
            insert_sql = '''
            INSERT INTO customers (
                customer_id, age, gender, country, location, monthly_spend,
                data_usage_gb, call_minutes, sms_count, roaming_usage,
                international_calls, premium_services, tenure_months,
                complaint_count, payment_method, satisfaction_score,
                current_plan, current_plan_price, upsell_revenue_potential
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(insert_sql, (
                row['customer_id'], row['age'], row['gender'], row['country'],
                row['location'], row['monthly_spend'], row['data_usage_gb'],
                row['call_minutes'], row['sms_count'], row['roaming_usage'],
                row['international_calls'], row['premium_services'],
                row['tenure_months'], row['complaint_count'], row['payment_method'],
                row['satisfaction_score'], row['current_plan'],
                row['current_plan_price'], row['upsell_revenue_potential']
            ))
        
        conn.commit()
        
        # Close connection appropriately
        if USE_AZURE_SQL or (DATABASE and DATABASE != ":memory:"):
            conn.close()
        
        print(f"Database initialized successfully with {len(df)} customers")
        
        # Train the model
        global model_trained, training_in_progress
        training_in_progress = True
        try:
            model_performance = predictor.train_model(df)
            model_trained = True
            print("Model trained successfully!")
            print(f"Model R² Score: {model_performance['r2_score']:.4f}")
        except Exception as e:
            print(f"Error training model: {e}")
            model_trained = False
        finally:
            training_in_progress = False
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()
        raise

def get_gpt_upsell_analysis(customer_data, prediction_result):
    """Get GPT analysis for upsell recommendations and planning"""
    if not GPT_API_ENABLED:
        return {
            'analysis': 'GPT analysis is currently disabled.',
            'recommendations': ['Enable GPT API for detailed recommendations'],
            'upsell_plan': 'GPT-powered upsell planning is not available.'
        }
    
    try:
        # Prepare customer context
        customer_context = f"""
        Customer Profile:
        - Age: {customer_data.get('age', 'N/A')}
        - Gender: {customer_data.get('gender', 'N/A')}
        - Country: {customer_data.get('country', 'N/A')}
        - Location: {customer_data.get('location', 'N/A')}
        - Current Plan: {customer_data.get('current_plan', 'N/A')} (QAR {customer_data.get('current_plan_price', 'N/A')}/month)
        - Monthly Spend: QAR {customer_data.get('monthly_spend', 'N/A')}
        - Data Usage: {customer_data.get('data_usage_gb', 'N/A')} GB
        - Call Minutes: {customer_data.get('call_minutes', 'N/A')}
        - SMS Count: {customer_data.get('sms_count', 'N/A')}
        - Roaming Usage: QAR {customer_data.get('roaming_usage', 'N/A')}
        - International Calls: {customer_data.get('international_calls', 'N/A')} minutes
        - Premium Services: QAR {customer_data.get('premium_services', 'N/A')}
        - Tenure: {customer_data.get('tenure_months', 'N/A')} months
        - Satisfaction Score: {customer_data.get('satisfaction_score', 'N/A')}/10
        - Complaint Count: {customer_data.get('complaint_count', 'N/A')}
        
        AI Prediction:
        - Predicted Revenue Increase Potential: QAR {prediction_result.get('predicted_revenue_increase', 'N/A')}
        - Confidence Score: {prediction_result.get('confidence_score', 'N/A')}
        """
        
        prompt = f"""
        You are an expert telecommunications analyst for Ooredoo, specializing in customer upselling strategies.
        
        {customer_context}
        
        Based on this customer profile and AI prediction, provide:
        
        1. CUSTOMER ANALYSIS (2-3 sentences):
        Analyze the customer's usage patterns, value potential, and behavioral indicators.
        
        2. UPSELL RECOMMENDATIONS (3-4 specific recommendations):
        Suggest specific Ooredoo services, plans, or add-ons that would benefit this customer.
        
        3. STRATEGIC UPSELL PLAN (detailed action plan):
        Create a step-by-step upsell strategy including:
        - Timing recommendations
        - Communication channels
        - Incentives or promotions
        - Risk mitigation strategies
        - Success metrics
        
        Focus on Ooredoo's actual services and consider the customer's country-specific market conditions.
        Keep recommendations practical and revenue-focused.
        """
        
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert telecom business analyst specializing in customer upselling strategies for Ooredoo."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        gpt_response = response.choices[0].message.content.strip()
        
        # Parse the response into sections
        sections = gpt_response.split('\n\n')
        analysis = ""
        recommendations = []
        upsell_plan = ""
        
        current_section = ""
        for section in sections:
            if "CUSTOMER ANALYSIS" in section.upper():
                current_section = "analysis"
                analysis = section.split(':', 1)[-1].strip()
            elif "UPSELL RECOMMENDATIONS" in section.upper():
                current_section = "recommendations"
            elif "STRATEGIC UPSELL PLAN" in section.upper() or "UPSELL PLAN" in section.upper():
                current_section = "plan"
                upsell_plan = section.split(':', 1)[-1].strip()
            elif current_section == "recommendations" and section.strip():
                # Extract bullet points or numbered items
                lines = section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 10))):
                        recommendations.append(line.lstrip('-•0123456789. '))
            elif current_section == "plan" and section.strip() and not upsell_plan:
                upsell_plan = section.strip()
        
        # Fallback parsing if structured parsing fails
        if not analysis and not recommendations and not upsell_plan:
            lines = gpt_response.split('\n')
            analysis = ' '.join(lines[:3])
            recommendations = [line.strip() for line in lines[3:7] if line.strip()]
            upsell_plan = ' '.join(lines[7:])
        
        return {
            'analysis': analysis or 'Customer shows potential for service upgrades based on usage patterns.',
            'recommendations': recommendations or ['Consider premium data plans', 'Explore international calling packages', 'Evaluate roaming add-ons'],
            'upsell_plan': upsell_plan or 'Implement targeted marketing campaign with personalized offers based on usage analysis.'
        }
        
    except Exception as e:
        print(f"Error getting GPT analysis: {e}")
        return {
            'analysis': f'Unable to generate AI analysis at this time. Error: {str(e)}',
            'recommendations': ['Review customer usage patterns manually', 'Consider standard upsell packages'],
            'upsell_plan': 'Manual analysis required for detailed upsell planning.'
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict upsell potential for a customer"""
    global predictor, model_trained
    
    if not model_trained:
        return jsonify({
            'error': 'Model not trained yet. Please wait for training to complete.'
        }), 400
    
    try:
        # Get customer data from request
        customer_data = request.json
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'country', 'location', 'monthly_spend',
            'data_usage_gb', 'call_minutes', 'sms_count', 'current_plan'
        ]
        
        for field in required_fields:
            if field not in customer_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set default values for optional fields
        defaults = {
            'roaming_usage': 0,
            'international_calls': 0,
            'premium_services': 0,
            'tenure_months': 12,
            'complaint_count': 0,
            'payment_method': 'Credit Card',
            'satisfaction_score': 7.5,
            'current_plan_price': 100
        }
        
        for key, value in defaults.items():
            if key not in customer_data:
                customer_data[key] = value
        
        # Make prediction
        prediction_result = predictor.predict_upsell_potential(customer_data)
        
        # Get GPT analysis
        gpt_analysis = get_gpt_upsell_analysis(customer_data, prediction_result)
        
        # Combine results
        result = {
            'prediction': prediction_result,
            'gpt_analysis': gpt_analysis,
            'customer_data': customer_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/customers')
def view_customers():
    """View all customers"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM customers LIMIT 100')
        customers = cursor.fetchall()
        
        # Get column names
        cursor.execute('PRAGMA table_info(customers)' if not USE_AZURE_SQL else 
                      "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'customers'")
        columns = [row[1] if not USE_AZURE_SQL else row[0] for row in cursor.fetchall()]
        
        # Close connection appropriately
        if USE_AZURE_SQL or (DATABASE and DATABASE != ":memory:"):
            conn.close()
        
        return render_template('customers.html', customers=customers, columns=columns)
        
    except Exception as e:
        print(f"Error viewing customers: {e}")
        return f"Error: {e}", 500

@app.route('/analytics')
def analytics_dashboard():
    """Customer Analytics Dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics-data')
def get_analytics_data():
    """Get analytics data for dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all customer data
        cursor.execute('SELECT * FROM customers')
        customers = cursor.fetchall()
        
        # Get column names
        cursor.execute('PRAGMA table_info(customers)' if not USE_AZURE_SQL else 
                      "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'customers'")
        columns = [row[1] if not USE_AZURE_SQL else row[0] for row in cursor.fetchall()]
        
        # Close connection appropriately
        if USE_AZURE_SQL or (DATABASE and DATABASE != ":memory:"):
            conn.close()
        
        if not customers:
            return jsonify({'error': 'No customer data available'}), 404
        
        # Convert to list of dictionaries
        customer_data = []
        for customer in customers:
            customer_dict = {}
            for i, column in enumerate(columns):
                customer_dict[column] = customer[i]
            customer_data.append(customer_dict)
        
        # Calculate statistics
        total_customers = len(customer_data)
        avg_revenue = sum(c.get('monthly_spend', 0) for c in customer_data) / total_customers if total_customers > 0 else 0
        avg_satisfaction = sum(c.get('satisfaction_score', 0) for c in customer_data) / total_customers if total_customers > 0 else 0
        high_potential_customers = sum(1 for c in customer_data if c.get('monthly_spend', 0) > 200)
        
        # Country distribution
        country_counts = {}
        for customer in customer_data:
            country = customer.get('country', 'Unknown')
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # Monthly spend distribution
        spend_ranges = {'0-50': 0, '51-100': 0, '101-200': 0, '201-300': 0, '301-500': 0, '500+': 0}
        for customer in customer_data:
            spend = customer.get('monthly_spend', 0)
            if spend <= 50:
                spend_ranges['0-50'] += 1
            elif spend <= 100:
                spend_ranges['51-100'] += 1
            elif spend <= 200:
                spend_ranges['101-200'] += 1
            elif spend <= 300:
                spend_ranges['201-300'] += 1
            elif spend <= 500:
                spend_ranges['301-500'] += 1
            else:
                spend_ranges['500+'] += 1
        
        # Age distribution
        age_ranges = {'18-25': 0, '26-35': 0, '36-45': 0, '46-55': 0, '56-65': 0, '65+': 0}
        for customer in customer_data:
            age = customer.get('age', 0)
            if age <= 25:
                age_ranges['18-25'] += 1
            elif age <= 35:
                age_ranges['26-35'] += 1
            elif age <= 45:
                age_ranges['36-45'] += 1
            elif age <= 55:
                age_ranges['46-55'] += 1
            elif age <= 65:
                age_ranges['56-65'] += 1
            else:
                age_ranges['65+'] += 1
        
        # Current plan distribution
        plan_counts = {}
        for customer in customer_data:
            plan = customer.get('current_plan', 'Unknown')
            plan_counts[plan] = plan_counts.get(plan, 0) + 1
        
        # Data usage vs revenue potential (scatter plot data)
        data_usage_revenue = []
        for customer in customer_data:
            data_usage = customer.get('data_usage_gb', 0)
            monthly_spend = customer.get('monthly_spend', 0)
            if data_usage is not None and monthly_spend is not None:
                data_usage_revenue.append({'x': float(data_usage), 'y': float(monthly_spend)})
        
        # Satisfaction vs complaints (scatter plot data)
        satisfaction_complaints = []
        for customer in customer_data:
            satisfaction = customer.get('satisfaction_score', 0)
            complaints = customer.get('complaint_count', 0)
            if satisfaction is not None and complaints is not None:
                satisfaction_complaints.append({'x': float(satisfaction), 'y': int(complaints)})
        
        # Prepare response data
        analytics_data = {
            'statistics': {
                'total_customers': total_customers,
                'avg_revenue': avg_revenue,
                'avg_satisfaction': avg_satisfaction,
                'high_potential_customers': high_potential_customers
            },
            'charts': {
                'country_distribution': {
                    'labels': list(country_counts.keys()),
                    'data': list(country_counts.values())
                },
                'spend_distribution': {
                    'labels': list(spend_ranges.keys()),
                    'data': list(spend_ranges.values())
                },
                'age_distribution': {
                    'labels': list(age_ranges.keys()),
                    'data': list(age_ranges.values())
                },
                'plan_distribution': {
                    'labels': list(plan_counts.keys()),
                    'data': list(plan_counts.values())
                },
                'data_usage_revenue': {
                    'data': data_usage_revenue
                },
                'satisfaction_complaints': {
                    'data': satisfaction_complaints
                }
            }
        }
        
        return jsonify(analytics_data)
        
    except Exception as e:
        print(f"Error getting analytics data: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-status')
def model_status():
    """Get model training status"""
    global model_trained, training_in_progress
    return jsonify({
        'trained': model_trained,
        'training_in_progress': training_in_progress
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model_trained,
        'database_type': 'Azure SQL' if USE_AZURE_SQL else 'SQLite',
        'gpt_enabled': GPT_API_ENABLED,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/initialize', methods=['POST'])
def manual_initialize():
    """Manually trigger database and model initialization"""
    global model_trained, training_in_progress
    
    if model_trained:
        return jsonify({
            'status': 'already_initialized',
            'message': 'Model is already trained and ready'
        })
    
    if training_in_progress:
        return jsonify({
            'status': 'training_in_progress',
            'message': 'Training is already in progress'
        })
    
    try:
        print("Manual initialization triggered...")
        init_database()
        return jsonify({
            'status': 'success',
            'message': 'Database and model initialized successfully',
            'model_trained': model_trained
        })
    except Exception as e:
        print(f"Error during manual initialization: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/api/force-reinitialize', methods=['POST'])
def force_reinitialize():
    """Force reinitialization of database and model (resets everything)"""
    global model_trained, training_in_progress, model
    
    try:
        print("Force reinitialization triggered...")
        
        # Reset model state
        model_trained = False
        training_in_progress = False
        model = None
        
        # Reinitialize everything
        init_database()
        
        return jsonify({
            'status': 'success',
            'message': 'Database and model force reinitialized successfully',
            'model_trained': model_trained
        })
    except Exception as e:
        print(f"Error during force reinitialization: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Force reinitialization failed: {str(e)}'
        }), 500

# Auto-initialize when module is imported (for Gunicorn)
initialize_app()

if __name__ == '__main__':
    try:
        print("Starting Upsell Revenue Predictor Application...")
        init_database()
        print("Application ready!")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    except Exception as e:
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        sys.exit(1)