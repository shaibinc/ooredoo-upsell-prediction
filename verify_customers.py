#!/usr/bin/env python3
"""
Simple script to verify customer count in database
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import get_db_connection, USE_AZURE_SQL

def verify_customer_count():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM customers')
        total_count = cursor.fetchone()[0]
        
        # Get recent customers (last 10)
        cursor.execute('SELECT customer_id, age, gender, country, monthly_spend FROM customers ORDER BY customer_id DESC LIMIT 10')
        recent_customers = cursor.fetchall()
        
        if USE_AZURE_SQL:
            conn.close()
        
        print(f"Database type: {'Azure SQL' if USE_AZURE_SQL else 'SQLite'}")
        print(f"Total customers in database: {total_count}")
        print("\nLast 10 customers added:")
        print("ID\tAge\tGender\tCountry\t\tSpend")
        print("-" * 50)
        
        for customer in recent_customers:
            cid, age, gender, country, spend = customer
            print(f"{cid}\t{age}\t{gender}\t{country[:10]}\t\t{spend:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_customer_count()