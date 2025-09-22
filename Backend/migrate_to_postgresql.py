#!/usr/bin/env python3
"""
PostgreSQL Migration Script for ARGO Float Data
Migrates data from SQLite to PostgreSQL with comprehensive error handling
"""

import os
import sys
import sqlite3
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class PostgreSQLMigration:
    def __init__(self):
        self.sqlite_path = "floatchat.db"
        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'database': os.getenv('PG_DATABASE', 'argo_floats'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'password'),
            'port': os.getenv('PG_PORT', '5432')
        }
        
    def test_connections(self):
        """Test both SQLite and PostgreSQL connections"""
        logger.info("🔍 Testing database connections...")
        
        # Test SQLite
        try:
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            cursor = sqlite_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM argo_measurements")
            sqlite_count = cursor.fetchone()[0]
            sqlite_conn.close()
            logger.info(f"✅ SQLite connection successful: {sqlite_count:,} records")
        except Exception as e:
            logger.error(f"❌ SQLite connection failed: {e}")
            return False
            
        # Test PostgreSQL
        try:
            pg_conn = psycopg2.connect(**self.pg_config)
            cursor = pg_conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            pg_conn.close()
            logger.info(f"✅ PostgreSQL connection successful: {version[:50]}...")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            logger.info("💡 Make sure PostgreSQL is installed and running")
            logger.info(f"💡 Create database: CREATE DATABASE {self.pg_config['database']};")
            return False
            
        return True
    
    def create_postgresql_schema(self):
        """Create PostgreSQL schema with proper data types"""
        logger.info("📊 Creating PostgreSQL schema...")
        
        schema_sql = """
        -- Drop table if exists (for clean migration)
        DROP TABLE IF EXISTS argo_measurements CASCADE;
        
        -- Create table with proper PostgreSQL data types
        CREATE TABLE argo_measurements (
            id SERIAL PRIMARY KEY,
            platform_number VARCHAR(20),
            cycle_number INTEGER,
            latitude DECIMAL(10,6),
            longitude DECIMAL(10,6),
            pressure DECIMAL(10,3),
            temperature DECIMAL(10,6),
            salinity DECIMAL(10,6),
            measurement_date TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX idx_argo_platform ON argo_measurements(platform_number);
        CREATE INDEX idx_argo_location ON argo_measurements(latitude, longitude);
        CREATE INDEX idx_argo_date ON argo_measurements(measurement_date);
        CREATE INDEX idx_argo_pressure ON argo_measurements(pressure);
        CREATE INDEX idx_argo_temperature ON argo_measurements(temperature);
        
        -- Create a view for quick statistics
        CREATE OR REPLACE VIEW argo_stats AS
        SELECT 
            COUNT(*) as total_measurements,
            COUNT(DISTINCT platform_number) as unique_platforms,
            MIN(latitude) as min_latitude,
            MAX(latitude) as max_latitude,
            MIN(longitude) as min_longitude,
            MAX(longitude) as max_longitude,
            MIN(temperature) as min_temperature,
            MAX(temperature) as max_temperature,
            AVG(temperature) as avg_temperature,
            MIN(pressure) as min_pressure,
            MAX(pressure) as max_pressure
        FROM argo_measurements;
        """
        
        try:
            pg_conn = psycopg2.connect(**self.pg_config)
            cursor = pg_conn.cursor()
            cursor.execute(schema_sql)
            pg_conn.commit()
            pg_conn.close()
            logger.info("✅ PostgreSQL schema created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False
    
    def migrate_data(self, batch_size=1000):
        """Migrate data in batches for better performance"""
        logger.info("🚀 Starting data migration...")
        
        # Connect to both databases
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        pg_conn = psycopg2.connect(**self.pg_config)
        
        try:
            # Get total count for progress tracking
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT COUNT(*) FROM argo_measurements")
            total_records = sqlite_cursor.fetchone()[0]
            logger.info(f"📈 Migrating {total_records:,} records in batches of {batch_size}")
            
            # Prepare PostgreSQL insert statement
            pg_cursor = pg_conn.cursor()
            insert_sql = """
                INSERT INTO argo_measurements 
                (platform_number, cycle_number, latitude, longitude, pressure, temperature, salinity, measurement_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Process data in batches
            offset = 0
            migrated_count = 0
            
            while offset < total_records:
                # Fetch batch from SQLite
                sqlite_cursor.execute(
                    "SELECT * FROM argo_measurements ORDER BY id LIMIT ? OFFSET ?",
                    (batch_size, offset)
                )
                rows = sqlite_cursor.fetchall()
                
                if not rows:
                    break
                
                # Prepare batch data for PostgreSQL
                batch_data = []
                for row in rows:
                    batch_data.append((
                        row['platform_number'],
                        row['cycle_number'], 
                        row['latitude'],
                        row['longitude'],
                        row['pressure'],
                        row['temperature'],
                        row['salinity'],
                        row['measurement_date']
                    ))
                
                # Insert batch into PostgreSQL
                pg_cursor.executemany(insert_sql, batch_data)
                pg_conn.commit()
                
                migrated_count += len(batch_data)
                offset += batch_size
                
                # Progress update
                progress = (migrated_count / total_records) * 100
                logger.info(f"📊 Progress: {migrated_count:,}/{total_records:,} ({progress:.1f}%)")
            
            logger.info(f"✅ Migration completed: {migrated_count:,} records migrated")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            pg_conn.rollback()
            return False
        finally:
            sqlite_conn.close()
            pg_conn.close()
            
        return True
    
    def verify_migration(self):
        """Verify that migration was successful"""
        logger.info("🔍 Verifying migration...")
        
        try:
            # Check SQLite count
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT COUNT(*) FROM argo_measurements")
            sqlite_count = sqlite_cursor.fetchone()[0]
            sqlite_conn.close()
            
            # Check PostgreSQL count
            pg_conn = psycopg2.connect(**self.pg_config)
            pg_cursor = pg_conn.cursor()
            pg_cursor.execute("SELECT COUNT(*) FROM argo_measurements")
            pg_count = pg_cursor.fetchone()[0]
            
            # Get some statistics
            pg_cursor.execute("SELECT * FROM argo_stats")
            stats = pg_cursor.fetchone()
            pg_conn.close()
            
            logger.info(f"📊 Verification Results:")
            logger.info(f"   SQLite records: {sqlite_count:,}")
            logger.info(f"   PostgreSQL records: {pg_count:,}")
            logger.info(f"   Migration success: {pg_count == sqlite_count}")
            
            if stats:
                logger.info(f"📈 PostgreSQL Statistics:")
                logger.info(f"   Unique platforms: {stats[1]}")
                logger.info(f"   Temperature range: {stats[7]:.2f}°C to {stats[8]:.2f}°C")
                logger.info(f"   Average temperature: {stats[9]:.2f}°C")
                logger.info(f"   Pressure range: {stats[10]:.1f} to {stats[11]:.1f} dbar")
            
            return pg_count == sqlite_count
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def run_migration(self):
        """Run the complete migration process"""
        logger.info("🚀 Starting PostgreSQL Migration for ARGO Float Data")
        logger.info("=" * 60)
        
        # Step 1: Test connections
        if not self.test_connections():
            logger.error("❌ Connection test failed. Aborting migration.")
            return False
        
        # Step 2: Create schema
        if not self.create_postgresql_schema():
            logger.error("❌ Schema creation failed. Aborting migration.")
            return False
        
        # Step 3: Migrate data
        if not self.migrate_data():
            logger.error("❌ Data migration failed. Aborting.")
            return False
        
        # Step 4: Verify migration
        if not self.verify_migration():
            logger.error("❌ Migration verification failed.")
            return False
        
        logger.info("=" * 60)
        logger.info("🎉 PostgreSQL Migration Completed Successfully!")
        logger.info("💡 Update your .env file:")
        logger.info("   USE_POSTGRESQL=true")
        logger.info(f"   DATABASE_URL=postgresql://{self.pg_config['user']}:***@{self.pg_config['host']}:{self.pg_config['port']}/{self.pg_config['database']}")
        
        return True

def main():
    """Main migration function"""
    migration = PostgreSQLMigration()
    
    print("🐘 PostgreSQL Migration Tool for ARGO Float Data")
    print("=" * 50)
    
    # Check if user wants to proceed
    response = input("This will migrate data from SQLite to PostgreSQL. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    success = migration.run_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("You can now use PostgreSQL for better performance and scalability.")
    else:
        print("\n❌ Migration failed. Check the logs above for details.")
        print("Your SQLite database remains unchanged.")

if __name__ == "__main__":
    main()