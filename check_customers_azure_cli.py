#!/usr/bin/env python3

import subprocess
import json
import sys

def run_azure_sql_query(query):
    """Run SQL query using Azure CLI"""
    try:
        cmd = [
            'az', 'sql', 'db', 'query',
            '--server', 'ooredoo-server',
            '--database', 'ooredoo-db',
            '--resource-group', 'ooredoo-upsell-rg-westus2',
            '--admin-user', 'CloudSA24079024',
            '--admin-password', 'Ooredoo@2024',
            '--query-text', query
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Error running query: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    print("=" * 60)
    print("Azure SQL Database Customer Verification (via Azure CLI)")
    print("=" * 60)
    print("Server: ooredoo-server.database.windows.net")
    print("Database: ooredoo-db")
    print()
    
    # Check if customers table exists
    print("Checking if customers table exists...")
    table_check = run_azure_sql_query(
        "SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'customers'"
    )
    
    if table_check and len(table_check) > 0:
        table_exists = table_check[0]['table_count'] > 0
        print(f"Customers table exists: {table_exists}")
        
        if table_exists:
            # Get total customer count
            print("\nGetting customer count...")
            count_result = run_azure_sql_query("SELECT COUNT(*) as total_customers FROM customers")
            
            if count_result and len(count_result) > 0:
                total_count = count_result[0]['total_customers']
                print(f"Total customers: {total_count}")
                
                if total_count > 0:
                    # Get sample customers
                    print("\nGetting sample customers...")
                    sample_result = run_azure_sql_query(
                        "SELECT TOP 5 customer_id, age, gender, country, monthly_spend, upsell_target FROM customers ORDER BY customer_id DESC"
                    )
                    
                    if sample_result:
                        print("\nSample customers:")
                        print("ID\tAge\tGender\tCountry\t\tSpend\tUpsell")
                        print("-" * 60)
                        for customer in sample_result:
                            print(f"{customer['customer_id']}\t{customer['age']}\t{customer['gender']}\t{customer['country'][:10]:<10}\t{customer['monthly_spend']:.2f}\t{customer['upsell_target']:.3f}")
                    
                    # Get country distribution
                    print("\nGetting country distribution...")
                    country_result = run_azure_sql_query(
                        "SELECT country, COUNT(*) as count FROM customers GROUP BY country ORDER BY count DESC"
                    )
                    
                    if country_result:
                        print("\nCustomers by country:")
                        for row in country_result[:10]:  # Top 10 countries
                            print(f"  {row['country']}: {row['count']} customers")
                else:
                    print("No customers found in the database.")
            else:
                print("Failed to get customer count.")
        else:
            print("Customers table does not exist. Database may not be initialized.")
    else:
        print("Failed to check table existence.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()