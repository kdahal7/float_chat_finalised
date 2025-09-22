
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Database Configuration with PostgreSQL primary, SQLite fallback
USE_POSTGRESQL = os.getenv("USE_POSTGRESQL", "true").lower() == "true"
POSTGRESQL_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/argo_floats")
SQLITE_URL = "sqlite:///floatchat.db"

# Choose database URL
if USE_POSTGRESQL:
    DATABASE_URL = POSTGRESQL_URL
    print("Using PostgreSQL database")
else:
    DATABASE_URL = SQLITE_URL
    print("Using SQLite database")

CSV_FILE_PATH = "processed_argo_data.csv"
ALL_DATA_FILE_PATH = "processed_all_argo_data.csv"

# Database connection with fallback mechanism
def create_database_engine():
    """Create database engine with PostgreSQL primary, SQLite fallback"""
    global DATABASE_URL, USE_POSTGRESQL
    
    if USE_POSTGRESQL:
        try:
            # Try PostgreSQL first
            engine = create_engine(
                POSTGRESQL_URL,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                echo=False
            )
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("PostgreSQL connection successful!")
            return engine
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            print("Falling back to SQLite...")
            USE_POSTGRESQL = False
            DATABASE_URL = SQLITE_URL
    
    # Use SQLite as fallback
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    print("SQLite connection successful!")
    return engine

# Create engine with fallback
engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get database session
def get_db_session():
    """Get database session for FastAPI dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ArgoMeasurement(Base):
    __tablename__ = "argo_measurements"
    
    # Use SERIAL for PostgreSQL, INTEGER AUTOINCREMENT for SQLite
    id = Column(Integer, primary_key=True, index=True)
    platform_number = Column(String(20), index=True)  # Specify length for PostgreSQL
    cycle_number = Column(Integer)
    latitude = Column(Float, index=True)  # DECIMAL in PostgreSQL, REAL in SQLite
    longitude = Column(Float, index=True)
    pressure = Column(Float)  # In decibars, proxy for depth
    temperature = Column(Float)
    salinity = Column(Float)
    measurement_date = Column(DateTime, index=True)

def setup_database():
    """Create database tables and return the engine."""
    try:
        # Create tables with proper PostgreSQL/SQLite compatibility
        Base.metadata.create_all(bind=engine)
        
        if USE_POSTGRESQL:
            print("PostgreSQL database tables created/verified.")
        else:
            print("SQLite database tables created/verified.")
        return engine
    except Exception as e:
        print(f"Database setup failed: {e}")
        return None

def get_database_info():
    """Get information about current database configuration"""
    return {
        "database_type": "PostgreSQL" if USE_POSTGRESQL else "SQLite",
        "database_url": DATABASE_URL,
        "fallback_available": True
    }

def store_argo_data_to_sqlite(engine_param=None):
    """Load data from the processed CSV into the SQLite database."""
    db_engine = engine_param or engine
    
    # Try to load the combined data file first, then fall back to original
    data_file = ALL_DATA_FILE_PATH if os.path.exists(ALL_DATA_FILE_PATH) else CSV_FILE_PATH
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        print("Please run the ARGO NetCDF processor and vectorstore pipeline first.")
        return False
    
    try:
        df = pd.read_csv(data_file)
        df = df.rename(columns={'date': 'measurement_date'})
        df['measurement_date'] = pd.to_datetime(df['measurement_date'], format='mixed')
        
        # Remove the 'source' column if it exists (added for tracking data sources)
        if 'source' in df.columns:
            df = df.drop('source', axis=1)
        
        # Check if table is empty before inserting
        with db_engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {ArgoMeasurement.__tablename__}"))
            count = result.scalar()
            if count > 0:
                # Clear existing data to reload with new NetCDF data
                print(f"Clearing existing {count} records to reload with updated data including NetCDF files...")
                connection.execute(text(f"DELETE FROM {ArgoMeasurement.__tablename__}"))
                connection.commit()

        print(f"Inserting {len(df)} records into the database. This may take a moment...")
        df.to_sql(ArgoMeasurement.__tablename__, db_engine, if_exists='append', index=False, chunksize=1000)
        print(f"✅ Successfully stored {len(df)} records to SQLite (including NetCDF data).")
        return True
        
    except Exception as e:
        print(f"Error storing data to SQLite: {e}")
        return False

if __name__ == "__main__":
    db_engine = setup_database()
    if db_engine:
        store_argo_data_to_sqlite(db_engine)