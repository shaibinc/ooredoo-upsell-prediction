#!/usr/bin/env python3

import pyodbc
import os

print("Available ODBC drivers:")
for driver in pyodbc.drivers():
    print(f"  {driver}")

print("\nTesting Azure SQL connection...")

# Test connection parameters
server = 'ooredoo-server.database.windows.net'
database = 'ooredoo-db'
username = 'CloudSA24079024'
password = 'Ooredoo@2024'

# Try different driver names
drivers_to_try = [
    'ODBC Driver 17 for SQL Server',
    'ODBC Driver 18 for SQL Server',
    'SQL Server'
]

for driver in drivers_to_try:
    try:
        print(f"\nTrying driver: {driver}")
        connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        print(f"Connection string: {connection_string}")
        
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute('SELECT COUNT(*) FROM customers')
        count = cursor.fetchone()[0]
        print(f"✅ Connection successful! Customer count: {count}")
        
        conn.close()
        break
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        continue
else:
    print("\n❌ All connection attempts failed")