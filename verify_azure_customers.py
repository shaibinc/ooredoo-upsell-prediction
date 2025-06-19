#!/usr/bin/env python3
"""
Script to verify customer count in Azure SQL Database
"""

import sys
import os

# Force Azure SQL usage by setting environment variables
os.environ['WEBSITE_SITE_NAME'] = 'ooredoo-upsell-app-2024'
os.environ['AZURE_SQL_SERVER'] = 'ooredoo-server.database.windows.net'
os.environ['AZURE_SQL_DATABASE'] = 'ooredoo-db'
os.environ['AZURE_SQL_USERNAME'] = 'CloudSA24079024'
os.environ['AZURE_SQL_PASSWORD'] = 'Ooredoo@2024'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pyodbc
import sqlite3

# Check if we should use Azure SQL
USE_AZURE_SQL = bool(os.environ.get('WEBSITE_SITE_NAME') or os.environ.get('AZURE_SQL_SERVER'))

def get_db_connection():
    if USE_AZURE_SQL:
        server = os.environ.get('AZURE_SQL_SERVER')
        database = os.environ.get('AZURE_SQL_DATABASE')
        username = os.environ.get('AZURE_SQL_USERNAME')
        password = os.environ.get('AZURE_SQL_PASSWORD')
        
        connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        return pyodbc.connect(connection_string)
    else:
        return sqlite3.connect('ooredoo_customers.db')

def verify_customer_count():
    try:
        print(f"Database mode: {'Azure SQL' if USE_AZURE_SQL else 'SQLite'}")
        print(f"Server: {os.environ.get('AZURE_SQL_SERVER')}")
        print(f"Database: {os.environ.get('AZURE_SQL_DATABASE')}")
        print()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM customers')
        total_count = cursor.fetchone()[0]
        
        # Get customers by country
        cursor.execute('SELECT country, COUNT(*) as count FROM customers GROUP BY country ORDER BY count DESC')
        country_stats = cursor.fetchall()
        
        # Get upsell distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN upsell_target < 0.4 THEN 'Low'
                    WHEN upsell_target >= 0.4 AND upsell_target < 0.7 THEN 'Medium'
                    ELSE 'High'
                END as upsell_category,
                COUNT(*) as count
            FROM customers 
            GROUP BY 
                CASE 
                    WHEN upsell_target < 0.4 THEN 'Low'
                    WHEN upsell_target >= 0.4 AND upsell_target < 0.7 THEN 'Medium'
                    ELSE 'High'
                END
            ORDER BY count DESC
        ''')
        upsell_stats = cursor.fetchall()
        
        # Get recent customers (last 10)
        if USE_AZURE_SQL:
            cursor.execute('''
                SELECT TOP 10 customer_id, age, gender, country, monthly_spend, upsell_target 
                FROM customers 
                ORDER BY customer_id DESC
            ''')
        else:
            cursor.execute('''
                SELECT customer_id, age, gender, country, monthly_spend, upsell_target 
                FROM customers 
                ORDER BY customer_id DESC
                LIMIT 10
            ''')
        recent_customers = cursor.fetchall()
        
        if USE_AZURE_SQL:
            conn.close()
        
        print(f"Total customers in Azure SQL Database: {total_count}")
        
        print("\nCustomers by country:")
        for country, count in country_stats:
            print(f"  {country}: {count} customers")
        
        print("\nUpsell potential distribution:")
        for category, count in upsell_stats:
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            print(f"  {category} potential: {count} customers ({percentage:.1f}%)")
        
        print("\nLast 10 customers added:")
        print("ID\tAge\tGender\tCountry\t\tSpend\tUpsell")
        print("-" * 60)
        
        for customer in recent_customers:
            cid, age, gender, country, spend, upsell = customer
            print(f"{cid}\t{age}\t{gender}\t{country[:10]:<10}\t{spend:.2f}\t{upsell:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Azure SQL Database Customer Verification")
    print("=" * 60)
    verify_customer_count()
    print("=" * 60)