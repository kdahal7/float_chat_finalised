#!/usr/bin/env python3
"""
Quick PostgreSQL Setup Test
Tests if PostgreSQL Python dependencies are working
"""

print("🧪 PostgreSQL Dependency Test")
print("=" * 40)

# Test 1: Check if psycopg2 is installed
print("1. Testing psycopg2 installation...")
try:
    import psycopg2
    print("   ✅ psycopg2 imported successfully")
    print(f"   Version: {psycopg2.__version__}")
except ImportError as e:
    print(f"   ❌ psycopg2 not found: {e}")
    print("   💡 Try: pip install psycopg2-binary")

# Test 2: Check if we can create a connection (will fail without server, but should not crash)
print("\n2. Testing PostgreSQL connection...")
try:
    import psycopg2
    # This will fail but shouldn't crash
    conn = psycopg2.connect(
        host='localhost',
        database='argo_floats', 
        user='postgres',
        password='password',
        port='5432',
        connect_timeout=1  # Quick timeout
    )
    print("   ✅ PostgreSQL server is running!")
    conn.close()
except psycopg2.OperationalError as e:
    print(f"   ⚠️ PostgreSQL server not running (expected): {str(e)[:50]}...")
    print("   💡 This is normal if PostgreSQL isn't installed yet")
except ImportError:
    print("   ❌ psycopg2 not available")
except Exception as e:
    print(f"   ⚠️ Other error: {e}")

# Test 3: Check our fallback mechanism
print("\n3. Testing fallback mechanism...")
try:
    import sys
    sys.path.append('.')
    
    # Import without reloading
    from database import USE_POSTGRESQL, get_database_info
    
    db_info = get_database_info()
    print(f"   Current database: {db_info['database_type']}")
    print(f"   USE_POSTGRESQL: {USE_POSTGRESQL}")
    print("   ✅ Fallback mechanism working")
    
except Exception as e:
    print(f"   ❌ Fallback test failed: {e}")

print("\n" + "=" * 40)
print("✅ Test completed!")
print("\n💡 Next steps:")
print("1. Install PostgreSQL server")
print("2. Run migration script")
print("3. Your system will automatically use PostgreSQL when available")