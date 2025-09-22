import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path
from robust_processor import process_argo_robust

# --- Configuration ---
CSV_FILE = "data/combined_argo_data.csv"
VECTORSTORE_DIR = "argo_faiss_vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 50000  # Limit number of rows to process for vector store to keep it fast

def process_all_netcdf_files():
    """Process all NetCDF files and return combined DataFrame"""
    print("🌊 Processing all NetCDF files...")
    
    # Look for NetCDF files in multiple locations
    nc_files = []
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        nc_files.extend(list(data_dir.glob("*.nc")))
    
    # Check current directory for additional NetCDF files
    current_dir = Path(".")
    nc_files.extend([f for f in current_dir.glob("*.nc") if f.name not in [ncf.name for ncf in nc_files]])
    
    print(f"Found NetCDF files: {[f.name for f in nc_files]}")
    
    all_netcdf_data = []
    for nc_file in nc_files:
        print(f"Processing: {nc_file.name}")
        df = process_argo_robust(nc_file)
        if df is not None:
            df['source'] = f'netcdf_{nc_file.name}'
            all_netcdf_data.append(df)
    
    if all_netcdf_data:
        combined_netcdf = pd.concat(all_netcdf_data, ignore_index=True)
        print(f"✅ Processed {len(combined_netcdf)} records from NetCDF files")
        return combined_netcdf
    else:
        print("⚠️ No data extracted from NetCDF files")
        return pd.DataFrame()

def load_and_combine_data():
    """Load existing CSV data and combine with NetCDF data"""
    all_dataframes = []
    
    # Load existing CSV if it exists
    if os.path.exists(CSV_FILE):
        print(f"📊 Loading existing CSV data from {CSV_FILE}")
        csv_df = pd.read_csv(CSV_FILE)
        csv_df['source'] = 'csv_existing'
        all_dataframes.append(csv_df)
        print(f"✅ Loaded {len(csv_df)} records from existing CSV")
    
    # Process NetCDF files
    netcdf_df = process_all_netcdf_files()
    if not netcdf_df.empty:
        all_dataframes.append(netcdf_df)
    
    # Combine all data
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"🔗 Combined total: {len(combined_df)} records from {len(all_dataframes)} sources")
        
        # Save updated combined data
        combined_df.to_csv("processed_all_argo_data.csv", index=False)
        print(f"💾 Saved all processed data to 'processed_all_argo_data.csv'")
        
        return combined_df
    else:
        print("❌ No data found in any source")
        return pd.DataFrame()

def create_documents_from_data(df):
    """Convert ARGO measurements into descriptive text documents for semantic search."""
    print(f"Creating documents from {len(df)} measurements...")
    documents = []
    for _, row in df.iterrows():
        text = (
            f"ARGO float {row['platform_number']} on {row['date']} at coordinates "
            f"({row['latitude']:.3f}, {row['longitude']:.3f}). "
            f"At a depth of {row['pressure']:.1f} dbar, the temperature was {row['temperature']:.2f}°C"
        )
        if pd.notna(row['salinity']):
            text += f" and salinity was {row['salinity']:.2f} PSU."
        
        metadata = {
            'platform_number': row['platform_number'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'pressure': row['pressure'],
            'date': row['date']
        }
        documents.append(Document(page_content=text, metadata=metadata))
    print(f"✅ Created {len(documents)} documents.")
    return documents

def main():
    """Main pipeline to create and save the FAISS vector store."""
    print("🌊 ARGO Vector Store Creation Pipeline")
    print("="*50)

    # Load and combine all data sources
    df = load_and_combine_data()
    
    if df.empty:
        print("❌ ERROR: No data available to process")
        return

    # Sample data if it's too large
    if len(df) > SAMPLE_SIZE:
        print(f"Dataset is large ({len(df)} records). Sampling {SAMPLE_SIZE} records for the vector store.")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    documents = create_documents_from_data(df)

    print(f"Initializing embedding model: {MODEL_NAME}")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    print("Creating FAISS vector store from documents. This may take some time...")
    vectorstore = FAISS.from_documents(documents, hf_embeddings)

    print(f"Saving vector store to directory: {VECTORSTORE_DIR}")
    vectorstore.save_local(VECTORSTORE_DIR)

    print("\n🎉 SUCCESS! ARGO Vector Store Created")
    print(f"✅ Processed data from NetCDF files and existing CSV")
    print(f"✅ Total documents in vector store: {len(documents)}")
    print(f"✅ Ready to use for the /api/semantic_query endpoint.")

if __name__ == "__main__":
    main()