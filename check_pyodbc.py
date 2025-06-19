#!/usr/bin/env python3
"""
Check if pyodbc is available for Azure SQL connections
"""

try:
    import pyodbc
    print("✅ pyodbc is available")
    print("Available ODBC drivers:")
    drivers = pyodbc.drivers()
    for driver in drivers:
        print(f"  - {driver}")
    
    # Check for SQL Server driver specifically
    sql_server_drivers = [d for d in drivers if 'SQL Server' in d]
    if sql_server_drivers:
        print(f"\n✅ SQL Server drivers found: {sql_server_drivers}")
    else:
        print("\n❌ No SQL Server drivers found")
        
except ImportError:
    print("❌ pyodbc is not installed")
    print("To install pyodbc, run: pip3 install pyodbc")
    print("Note: You may also need to install Microsoft ODBC Driver for SQL Server")