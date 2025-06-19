#!/usr/bin/env python3
"""
Script to generate 400 dummy customer records and insert them into Azure SQL Database
This script forces Azure SQL usage by setting environment variables
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Force Azure SQL usage by setting environment variables
os.environ['WEBSITE_SITE_NAME'] = 'ooredoo-upsell-app-2024'  # This triggers Azure mode
os.environ['AZURE_SQL_SERVER'] = 'ooredoo-server.database.windows.net'
os.environ['AZURE_SQL_DATABASE'] = 'ooredoo-db'
os.environ['AZURE_SQL_USERNAME'] = 'CloudSA24079024'
os.environ['AZURE_SQL_PASSWORD'] = 'Ooredoo@2024'

# Add current directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required functions and classes from app.py
from app import OoreedooUpsellPredictor, insert_customers_to_db, get_db_connection, USE_AZURE_SQL

def generate_and_insert_customers(num_customers=400):
    """
    Generate dummy customer data and insert into Azure SQL Database
    """
    print(f"Starting generation of {num_customers} dummy customers...")
    print(f"Database mode: {'Azure SQL' if USE_AZURE_SQL else 'SQLite'}")
    print(f"Server: {os.environ.get('AZURE_SQL_SERVER')}")
    print(f"Database: {os.environ.get('AZURE_SQL_DATABASE')}")
    
    try:
        # Test database connection first
        print("Testing database connection...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        existing_count = cursor.fetchone()[0]
        print(f"Current customer count in database: {existing_count}")
        
        if USE_AZURE_SQL:
            conn.close()
        
        # Create predictor instance
        print("Initializing Ooredoo Upsell Predictor...")
        predictor = OoreedooUpsellPredictor()
        
        # Generate sample data
        print(f"Generating {num_customers} customer records...")
        sample_data = predictor.generate_sample_data(num_customers)
        
        # Adjust customer IDs to avoid conflicts with existing data
        if existing_count > 0:
            # Get the maximum existing customer_id
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(customer_id) FROM customers")
            result = cursor.fetchone()[0]
            max_existing_id = result if result is not None else 0
            if USE_AZURE_SQL:
                conn.close()
            
            sample_data['customer_id'] = range(max_existing_id + 1, max_existing_id + num_customers + 1)
            print(f"Adjusted customer IDs to start from {max_existing_id + 1}")
        
        print("Sample data generated successfully!")
        print(f"Data shape: {sample_data.shape}")
        print("\nSample of generated data:")
        print(sample_data.head())
        
        # Insert data into database
        print(f"\nInserting {num_customers} customers into database...")
        rows_inserted = insert_customers_to_db(sample_data)
        
        print(f"Successfully inserted {rows_inserted} customers into the database!")
        
        # Verify insertion
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        new_count = cursor.fetchone()[0]
        if USE_AZURE_SQL:
            conn.close()
        
        print(f"Total customers in database after insertion: {new_count}")
        print(f"New customers added: {new_count - existing_count}")
        
        # Show distribution by country
        print("\nCustomer distribution by country:")
        country_dist = sample_data['country'].value_counts()
        for country, count in country_dist.items():
            print(f"  {country}: {count} customers")
        
        # Show upsell target distribution
        print("\nUpsell target distribution:")
        low_upsell = len(sample_data[sample_data['upsell_target'] < 0.4])
        medium_upsell = len(sample_data[(sample_data['upsell_target'] >= 0.4) & (sample_data['upsell_target'] < 0.7)])
        high_upsell = len(sample_data[sample_data['upsell_target'] >= 0.7])
        
        print(f"  Low potential (< 0.4): {low_upsell} customers ({low_upsell/num_customers*100:.1f}%)")
        print(f"  Medium potential (0.4-0.7): {medium_upsell} customers ({medium_upsell/num_customers*100:.1f}%)")
        print(f"  High potential (>= 0.7): {high_upsell} customers ({high_upsell/num_customers*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the customer generation script
    """
    print("=" * 60)
    print("Ooredoo Customer Data Generator (Azure SQL)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if custom number of customers is provided
    num_customers = 400
    if len(sys.argv) > 1:
        try:
            num_customers = int(sys.argv[1])
            if num_customers <= 0:
                print("Error: Number of customers must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid number of customers provided")
            sys.exit(1)
    
    print(f"Target: Generate {num_customers} dummy customers")
    print()
    
    # Generate and insert customers
    success = generate_and_insert_customers(num_customers)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Customer generation completed successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Customer generation failed!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()