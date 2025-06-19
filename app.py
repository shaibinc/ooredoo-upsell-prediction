from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from datetime import datetime
import sqlite3
import json
import warnings
import openai
import requests
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Database configuration
import tempfile
import os
# Use a temporary directory for the database in Azure
if os.environ.get('WEBSITE_SITE_NAME'):  # Running in Azure
    DATABASE = '/tmp/ooredoo_customers.db'
else:  # Running locally
    DATABASE = 'ooredoo_customers.db'
# Global database connection for in-memory database in Azure
_global_db_connection = None

def get_db_connection():
    """Get database connection, reusing global connection for in-memory database"""
    global _global_db_connection
    if DATABASE == ":memory:" and _global_db_connection:
        return _global_db_connection
    else:
        return sqlite3.connect(DATABASE)

# GPT API Configuration (Azure OpenAI)
# Note: In production, store these in environment variables or Azure Key Vault
GPT_API_ENABLED = True  # Enable the integration
OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = "https://upselloreedoo.openai.azure.com/"
OPENAI_API_VERSION = "2025-01-01-preview"
OPENAI_API_KEY = "3RnTuW94N3FjQQ3ZAf3OQuDVZnUGxBnj9lVrljA2PesMyMIOCLx6JQQJ99BFACYeBjFXJ3w3AAABACOGTvRr"
OPENAI_ENGINE = "gpt-4"

# Configure OpenAI if enabled
if GPT_API_ENABLED:
    openai.api_type = OPENAI_API_TYPE
    openai.api_base = OPENAI_API_BASE
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

def init_database():
    """Initialize the SQLite database"""
    import os
    import sqlite3
    import time
    import tempfile
    
    print("Attempting to initialize database...")
    
    # In Azure, use in-memory database to completely avoid file corruption issues
    if os.environ.get('WEBSITE_SITE_NAME'):  # Running in Azure
        global DATABASE
        DATABASE = ":memory:"
        print("Azure environment detected, using in-memory database to avoid corruption")
    else:
        # Local environment - remove existing database
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if os.path.exists(DATABASE):
                    print(f"Removing existing database: {DATABASE} (attempt {attempt + 1})")
                    os.remove(DATABASE)
                    time.sleep(0.1)  # Brief pause to ensure file is released
                    print("Existing database removed successfully")
                break
            except OSError as remove_error:
                print(f"Error removing existing database (attempt {attempt + 1}): {remove_error}")
                if attempt == max_retries - 1:
                    print("Failed to remove database after all attempts, continuing anyway...")
                time.sleep(0.5)
    
    max_db_retries = 3
    for db_attempt in range(max_db_retries):
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            print(f"Database connection established: {DATABASE} (attempt {db_attempt + 1})")
            
            # Store global connection for in-memory database
            if DATABASE == ":memory:":
                global _global_db_connection
                _global_db_connection = conn
            
            break
        except sqlite3.Error as e:
            print(f"Failed to connect to database (attempt {db_attempt + 1}): {e}")
            if db_attempt == max_db_retries - 1:
                raise
            time.sleep(1)
    
    # Create customers table
    try:
        cursor.execute('''
         CREATE TABLE IF NOT EXISTS customers (
             customer_id INTEGER PRIMARY KEY,
             age INTEGER,
             gender TEXT,
             country TEXT,
             location TEXT,
             monthly_spend REAL,
             data_usage_gb REAL,
             call_minutes INTEGER,
             sms_count INTEGER,
             tenure_months INTEGER,
             complaint_count INTEGER,
             payment_method TEXT,
             current_plan TEXT,
             upsell_target REAL,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
         )
         ''')
        conn.commit()
        print("Database table created successfully")
        
        # Check if table is empty and populate with sample data
        cursor.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        if count == 0:
            print("Database is empty, generating sample data...")
            # Create predictor instance and generate sample data
            predictor = OoreedooUpsellPredictor()
            sample_data = predictor.generate_sample_data(500)
            predictor.insert_customers_to_db(sample_data)
            print(f"Inserted {len(sample_data)} sample customers")
        else:
            print(f"Database contains {count} customers")
            
    except sqlite3.Error as e:
        print(f"Error creating database table: {e}")
        conn.rollback()
        raise
    finally:
        # Don't close connection for in-memory database
        if DATABASE != ":memory:":
            conn.close()
        print("Database initialization completed")

def get_gpt_explanation(customer_data, prediction_prob, recommendation):
    """Generate GPT-powered explanation and marketing recommendation"""
    if not GPT_API_ENABLED:
        # Fallback explanations when GPT API is not available
        fallback_explanations = {
            'High': {
                'explanation': 'This customer shows strong indicators for upsell acceptance based on high data usage, spending patterns, and engagement history.',
                'marketing_suggestion': 'Offer premium data packages or value-added services with personalized benefits.'
            },
            'Medium': {
                'explanation': 'This customer has moderate potential for upsell acceptance with some positive indicators in their usage patterns.',
                'marketing_suggestion': 'Present targeted offers with clear value propositions and limited-time incentives.'
            },
            'Low': {
                'explanation': 'This customer shows limited indicators for upsell acceptance based on current usage and spending patterns.',
                'marketing_suggestion': 'Focus on retention strategies and gradual engagement with small, attractive offers.'
            }
        }
        return fallback_explanations.get(recommendation, fallback_explanations['Medium'])
    
    try:
        # Prepare customer profile summary
        profile_summary = (
            f"Age: {customer_data.get('age', 'N/A')}, "
            f"Gender: {customer_data.get('gender', 'N/A')}, "
            f"Country: {customer_data.get('country', 'N/A')}, "
            f"Monthly Spend: ${customer_data.get('monthly_spend', 0):.2f}, "
            f"Data Usage: {customer_data.get('data_usage_gb', 0):.1f}GB, "
            f"Call Minutes: {customer_data.get('call_minutes', 0)}, "
            f"Tenure: {customer_data.get('tenure_months', 0)} months, "
            f"Complaints: {customer_data.get('complaint_count', 0)}, "
            f"Payment Method: {customer_data.get('payment_method', 'N/A')}, "
            f"Current Plan: {customer_data.get('current_plan', 'N/A')}"
        )
        
        # Create GPT prompt
        prompt = (
            f"As a telecom marketing expert for Ooredoo, analyze this customer profile: {profile_summary}. "
            f"Our AI model predicts a {prediction_prob*100:.1f}% likelihood of accepting a mobile service upsell. "
            f"Provide: 1) A brief explanation (max 50 words) of why this prediction makes sense based on the customer's behavior, "
            f"and 2) A specific, actionable marketing recommendation (max 40 words) for this customer. "
            f"Format your response as: EXPLANATION: [your explanation] RECOMMENDATION: [your recommendation]"
        )
        
        # Make API call to GPT
        response = openai.ChatCompletion.create(
            engine=OPENAI_ENGINE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
            timeout=10
        )
        
        gpt_response = response.choices[0].message['content'].strip()
        
        # Parse the response
        explanation = ""
        marketing_suggestion = ""
        
        if "EXPLANATION:" in gpt_response and "RECOMMENDATION:" in gpt_response:
            parts = gpt_response.split("RECOMMENDATION:")
            explanation = parts[0].replace("EXPLANATION:", "").strip()
            marketing_suggestion = parts[1].strip()
        else:
            # Fallback if parsing fails
            explanation = gpt_response[:100] + "..." if len(gpt_response) > 100 else gpt_response
            marketing_suggestion = "Contact customer with personalized offer based on their usage patterns."
        
        return {
            'explanation': explanation,
            'marketing_suggestion': marketing_suggestion
        }
        
    except Exception as e:
        print(f"GPT API Error: {str(e)}")
        # Return fallback explanation on error
        fallback_explanations = {
            'High': {
                'explanation': 'Customer shows strong upsell potential based on usage patterns and engagement metrics.',
                'marketing_suggestion': 'Offer premium services with personalized benefits and exclusive deals.'
            },
            'Medium': {
                'explanation': 'Customer has moderate upsell potential with some positive behavioral indicators.',
                'marketing_suggestion': 'Present targeted offers with clear value propositions and incentives.'
            },
            'Low': {
                'explanation': 'Customer shows limited upsell potential based on current usage and spending patterns.',
                'marketing_suggestion': 'Focus on retention with small, attractive offers to build engagement.'
            }
        }
        return fallback_explanations.get(recommendation, fallback_explanations['Medium'])

def insert_customers_to_db(customers_df):
    """Insert customer data into the database"""
    conn = get_db_connection()
    
    # Convert DataFrame to list of tuples for insertion
    customers_data = []
    for _, row in customers_df.iterrows():
        customers_data.append((
            row['customer_id'],
            int(row['age']),
            row['gender'],
            row['country'],
            row['location'],
            float(row['monthly_spend']),
            float(row['data_usage_gb']),
            int(row['call_minutes']),
            int(row['sms_count']),
            int(row['tenure_months']),
            int(row['complaint_count']),
            row['payment_method'],
            row['current_plan'],
            float(row['upsell_target'])
        ))
    
    cursor = conn.cursor()
    
    # Insert customers (ignore duplicates)
    cursor.executemany('''
        INSERT OR IGNORE INTO customers 
        (customer_id, age, gender, country, location, monthly_spend, data_usage_gb, 
         call_minutes, sms_count, tenure_months, complaint_count, 
         payment_method, current_plan, upsell_target)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', customers_data)
    
    conn.commit()
    rows_inserted = cursor.rowcount
    # Don't close connection for in-memory database
    if DATABASE != ":memory:":
        conn.close()
    
    return rows_inserted

def get_customers_from_db(limit=None):
    """Retrieve customers from the database"""
    conn = get_db_connection()
    
    query = '''
        SELECT customer_id, age, gender, country, location, monthly_spend, data_usage_gb,
               call_minutes, sms_count, tenure_months, complaint_count,
               payment_method, current_plan, upsell_target, created_at
        FROM customers
        ORDER BY created_at DESC
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    df = pd.read_sql_query(query, conn)
    # Don't close connection for in-memory database
    if DATABASE != ":memory:":
        conn.close()
    
    return df

def get_customer_count():
    """Get total number of customers in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM customers')
    count = cursor.fetchone()[0]
    
    # Don't close connection for in-memory database
    if DATABASE != ":memory:":
        conn.close()
    return count

# Global variables for model and encoders
model = None
label_encoders = {}
scaler = None
feature_columns = []

class OoreedooUpsellPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def generate_sample_data(self, n_samples=500):
        """Generate sample customer data for Ooredoo"""
        np.random.seed(42)
        
        # Customer demographics
        customer_ids = list(range(1, n_samples + 1))
        ages = np.random.normal(35, 12, n_samples).astype(int)
        ages = np.clip(ages, 18, 80)
        genders = np.random.choice(['Male', 'Female'], n_samples)
        # Ooredoo operates in 9 countries
        countries = np.random.choice(['Qatar', 'Oman', 'Kuwait', 'Algeria', 'Tunisia', 'Iraq', 'Palestine', 'Maldives', 'Myanmar'], n_samples)
        
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
        
        # Generate locations based on countries
        locations = []
        for country in countries:
            location = np.random.choice(country_locations[country])
            locations.append(location)
        locations = np.array(locations)
        
        # Service usage patterns (QAR - Qatar Riyal)
        monthly_spend = np.random.exponential(200, n_samples) + 50  # QAR (adjusted for Qatar market)
        data_usage_gb = np.random.exponential(15, n_samples) + 5
        call_minutes = np.random.exponential(200, n_samples) + 50
        sms_count = np.random.exponential(100, n_samples) + 10
        
        # Customer behavior
        tenure_months = np.random.exponential(24, n_samples) + 1
        complaint_count = np.random.poisson(0.5, n_samples)
        payment_method = np.random.choice(['Credit Card', 'Bank Transfer', 'Cash', 'Mobile Payment'], n_samples)
        
        # Current plan types
        current_plans = np.random.choice([
            'Hala Super 40', 'Hala Digital', 'Hala Visitor SIM',
            'Shahry+ Select', 'Shahry+ Go', 'Qatarna+ Pro', 
            'Qatarna+ Premium', 'Qatarna+ Al Nokhba'
        ], n_samples)
        
        # Country-specific upsell factors (based on market maturity and purchasing power)
        country_factors = {
            'Qatar': 1.2, 'Kuwait': 1.1, 'Oman': 1.0, 'Algeria': 0.8, 
            'Tunisia': 0.8, 'Iraq': 0.7, 'Palestine': 0.6, 'Maldives': 0.9, 'Myanmar': 0.7
        }
        country_multipliers = np.array([country_factors[country] for country in countries])
        
        # Create balanced distribution: Low (0.0-0.4), Medium (0.4-0.7), High (0.7-1.0)
        # Allocate customers to segments: 35% Low, 35% Medium, 30% High
        segment_sizes = [int(n_samples * 0.35), int(n_samples * 0.35), int(n_samples * 0.30)]
        # Adjust for rounding
        segment_sizes[2] = n_samples - sum(segment_sizes[:2])
        
        upsell_target = np.zeros(n_samples)
        current_idx = 0
        
        # Low upsell segment (0.05 - 0.35)
        low_end = current_idx + segment_sizes[0]
        upsell_target[current_idx:low_end] = np.random.uniform(0.05, 0.35, segment_sizes[0])
        current_idx = low_end
        
        # Medium upsell segment (0.40 - 0.65)
        med_end = current_idx + segment_sizes[1]
        upsell_target[current_idx:med_end] = np.random.uniform(0.40, 0.65, segment_sizes[1])
        current_idx = med_end
        
        # High upsell segment (0.70 - 0.95)
        upsell_target[current_idx:] = np.random.uniform(0.70, 0.95, segment_sizes[2])
        
        # Apply country-specific multipliers with constraints
        upsell_target = upsell_target * country_multipliers
        
        # Add feature-based adjustments (smaller impact to maintain distribution)
        feature_adjustment = (
            (monthly_spend / 1000) * 0.1 +     # Spend factor
            (data_usage_gb / 100) * 0.08 +     # Data usage factor
            (call_minutes / 2000) * 0.05 +     # Call factor
            (tenure_months / 100) * 0.05 +     # Tenure factor
            ((5 - complaint_count) / 10) * 0.02 # Satisfaction factor
        )
        
        upsell_target += feature_adjustment
        
        # Shuffle to randomize order and ensure final bounds
        np.random.shuffle(upsell_target)
        upsell_target = np.clip(upsell_target, 0, 1)
        
        data = {
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'country': countries,
            'location': locations,
            'monthly_spend': monthly_spend,  # QAR
            'data_usage_gb': data_usage_gb,
            'call_minutes': call_minutes,
            'sms_count': sms_count,
            'tenure_months': tenure_months,
            'complaint_count': complaint_count,
            'payment_method': payment_method,
            'current_plan': current_plans,
            'upsell_target': upsell_target
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for machine learning"""
        df_processed = df.copy()
        
        # Remove customer_id for modeling
        if 'customer_id' in df_processed.columns:
            df_processed = df_processed.drop('customer_id', axis=1)
        
        # Encode categorical variables
        categorical_columns = ['gender', 'country', 'location', 'payment_method', 'current_plan']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df_processed[col].unique())
                        known_values = set(self.label_encoders[col].classes_)
                        
                        for val in unique_values - known_values:
                            df_processed[col] = df_processed[col].replace(val, self.label_encoders[col].classes_[0])
                        
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train_model(self, df):
        """Train the upsell prediction model"""
        # Preprocess data
        df_processed = self.preprocess_data(df, is_training=True)
        
        # Separate features and target
        X = df_processed.drop('upsell_target', axis=1)
        y = df_processed['upsell_target']
        
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (no stratify for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        
        return r2
    
    def predict_upsell(self, customer_data):
        """Predict upsell probability for a customer with GPT explanations"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
            original_data = customer_data.copy()
        else:
            df = customer_data.copy()
            original_data = df.iloc[0].to_dict() if len(df) > 0 else {}
        
        # Preprocess data
        df_processed = self.preprocess_data(df, is_training=False)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Reorder columns to match training data
        df_processed = df_processed[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Make prediction (regression output is already a probability)
        probability = self.model.predict(X_scaled)[0]
        
        # Ensure probability is between 0 and 1
        probability = np.clip(probability, 0, 1)
        
        # Convert to binary prediction for compatibility
        prediction = 1 if probability > 0.5 else 0
        
        # Determine recommendation level
        recommendation = 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
        
        # Get GPT explanation and marketing suggestion
        gpt_insights = get_gpt_explanation(original_data, probability, recommendation)
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'recommendation': recommendation,
            'explanation': gpt_insights['explanation'],
            'marketing_suggestion': gpt_insights['marketing_suggestion']
        }
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model and encoders"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']

# Initialize database and predictor
predictor = None

try:
    print("Initializing database...")
    init_database()
    print("Database initialized successfully!")
except Exception as e:
    print(f"Database initialization failed: {e}")
    print("App will continue, database will be initialized on first request")
    # Ensure we don't crash the app due to database issues
    pass

try:
    predictor = OoreedooUpsellPredictor()
    
    # Train model on startup
    if not os.path.exists('ooredoo_model.pkl'):
        print("Training new model...")
        sample_data = predictor.generate_sample_data(2000)
        predictor.train_model(sample_data)
        predictor.save_model('ooredoo_model.pkl')
        print("Model trained and saved!")
    else:
        print("Loading existing model...")
        predictor.load_model('ooredoo_model.pkl')
        print("Model loaded successfully!")
    
    # Check if we need to populate the database with sample data
    try:
        customer_count = get_customer_count()
        if customer_count == 0:
            print("Database is empty. Generating 500 sample customers...")
            sample_customers = predictor.generate_sample_data(500)
            rows_inserted = insert_customers_to_db(sample_customers)
            print(f"Inserted {rows_inserted} customers into the database!")
        else:
            print(f"Database already contains {customer_count} customers.")
    except Exception as e:
        print(f"Failed to populate database: {e}")
        print("Database will be populated on first request")
        
except Exception as e:
    print(f"Model initialization failed: {e}")
    print("Model will be initialized on first request")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predictor
    try:
        # Initialize predictor if not already done
        if predictor is None:
            print("Lazy initializing predictor...")
            predictor = OoreedooUpsellPredictor()
            if os.path.exists('ooredoo_model.pkl'):
                predictor.load_model('ooredoo_model.pkl')
            else:
                sample_data = predictor.generate_sample_data(2000)
                predictor.train_model(sample_data)
                predictor.save_model('ooredoo_model.pkl')
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['age', 'gender', 'country', 'location', 'monthly_spend', 
                          'data_usage_gb', 'call_minutes', 'sms_count', 
                          'tenure_months', 'complaint_count', 'payment_method', 'current_plan']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = predictor.predict_upsell(data)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/customer_analysis')
def customer_analysis():
    return render_template('customer_analysis.html')

@app.route('/database')
def database_viewer():
    return render_template('database_viewer.html')

@app.route('/api/sample_data')
def get_sample_data():
    """Generate sample data for testing"""
    sample = {
        'age': 32,
        'gender': 'Male',
        'country': 'Qatar',
        'location': 'Doha',
        'monthly_spend': 75.5,
        'data_usage_gb': 25.2,
        'call_minutes': 450,
        'sms_count': 120,
        'tenure_months': 18,
        'complaint_count': 0,
        'payment_method': 'Credit Card',
        'current_plan': 'Shahry+ Select'
    }
    return jsonify(sample)

@app.route('/api/customers')
def get_customers():
    """Get all customers from database"""
    try:
        limit = request.args.get('limit', type=int)
        customers_df = get_customers_from_db(limit=limit)
        
        # Convert DataFrame to list of dictionaries
        customers_list = customers_df.to_dict('records')
        
        return jsonify({
            'customers': customers_list,
            'total_count': len(customers_list),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers/generate', methods=['POST'])
def generate_sample_customers():
    """Generate and insert 100 new sample customers"""
    try:
        # Generate 100 new sample customers
        sample_customers = predictor.generate_sample_data(100)
        
        # Insert into database
        rows_inserted = insert_customers_to_db(sample_customers)
        
        return jsonify({
            'message': f'Successfully generated and inserted {rows_inserted} new customers',
            'customers_inserted': rows_inserted,
            'total_customers': get_customer_count(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers/stats')
def get_customer_stats():
    """Get customer statistics from database"""
    try:
        customers_df = get_customers_from_db()
        
        if customers_df.empty:
            return jsonify({
                'total_customers': 0,
                'message': 'No customers in database'
            })
        
        # Calculate statistics
        stats = {
            'total_customers': len(customers_df),
            'high_potential': len(customers_df[customers_df['upsell_target'] == 1]),
            'low_potential': len(customers_df[customers_df['upsell_target'] == 0]),
            'avg_monthly_spend': float(customers_df['monthly_spend'].mean()),
            'avg_data_usage': float(customers_df['data_usage_gb'].mean()),
            'avg_tenure': float(customers_df['tenure_months'].mean()),
            'plan_distribution': customers_df['current_plan'].value_counts().to_dict(),
            'location_distribution': customers_df['location'].value_counts().to_dict(),
            'gender_distribution': customers_df['gender'].value_counts().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers/predict-batch', methods=['POST'])
def predict_batch():
    """Predict upsell probability for all customers in database"""
    try:
        customers_df = get_customers_from_db()
        
        if customers_df.empty:
            return jsonify({'error': 'No customers in database'}), 400
        
        predictions = []
        
        for _, customer in customers_df.iterrows():
            customer_data = {
                'age': customer['age'],
                'gender': customer['gender'],
                'location': customer['location'],
                'monthly_spend': customer['monthly_spend'],
                'data_usage_gb': customer['data_usage_gb'],
                'call_minutes': customer['call_minutes'],
                'sms_count': customer['sms_count'],
                'tenure_months': customer['tenure_months'],
                'complaint_count': customer['complaint_count'],
                'payment_method': customer['payment_method'],
                'current_plan': customer['current_plan']
            }
            
            result = predictor.predict_upsell(customer_data)
            result['customer_id'] = customer['customer_id']
            predictions.append(result)
        
        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'predictions': predictions,
            'total_customers': len(predictions),
            'high_priority': len([p for p in predictions if p['recommendation'] == 'High']),
            'medium_priority': len([p for p in predictions if p['recommendation'] == 'Medium']),
            'low_priority': len([p for p in predictions if p['recommendation'] == 'Low']),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'database_customers': get_customer_count(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Azure App Service configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)