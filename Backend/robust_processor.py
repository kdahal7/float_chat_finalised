import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def convert_julian_date(juld):
    """Convert ARGO Julian date to datetime string."""
    try:
        if pd.isna(juld) or juld <= 0 or np.isnan(juld):
            return None
        reference_date = pd.to_datetime('1950-01-01')
        result_date = reference_date + pd.to_timedelta(float(juld), unit='D')
        return result_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError, OverflowError):
        return None

def smart_variable_detection(ds):
    """Smart detection of ARGO variables with multiple naming conventions"""
    var_map = {}
    
    # Latitude detection - prioritize standard names
    if 'LATITUDE' in ds.variables:
        var_map['lat'] = 'LATITUDE'
    elif 'latitude' in ds.variables:
        var_map['lat'] = 'latitude'
    else:
        lat_candidates = [v for v in ds.variables if any(pattern in v.lower() for pattern in ['lat'])]
        var_map['lat'] = lat_candidates[0] if lat_candidates else None
    
    # Longitude detection - prioritize standard names
    if 'LONGITUDE' in ds.variables:
        var_map['lon'] = 'LONGITUDE'
    elif 'longitude' in ds.variables:
        var_map['lon'] = 'longitude'
    else:
        lon_candidates = [v for v in ds.variables if any(pattern in v.lower() for pattern in ['lon'])]
        var_map['lon'] = lon_candidates[0] if lon_candidates else None
    
    # Pressure detection - prioritize actual data over QC flags
    if 'PRES' in ds.variables:
        var_map['pres'] = 'PRES'
    elif 'pres' in ds.variables:
        var_map['pres'] = 'pres'
    elif 'pressure' in ds.variables:
        var_map['pres'] = 'pressure'
    else:
        pres_candidates = [v for v in ds.variables if ('pres' in v.lower() or 'depth' in v.lower()) and '_qc' not in v.lower()]
        var_map['pres'] = pres_candidates[0] if pres_candidates else None
    
    # Temperature detection - prioritize actual data over QC flags
    if 'TEMP' in ds.variables:
        var_map['temp'] = 'TEMP'
    elif 'temp' in ds.variables:
        var_map['temp'] = 'temp'
    elif 'temperature' in ds.variables:
        var_map['temp'] = 'temperature'
    else:
        temp_candidates = [v for v in ds.variables if 'temp' in v.lower() and '_qc' not in v.lower()]
        var_map['temp'] = temp_candidates[0] if temp_candidates else None
    
    # Salinity detection - prioritize actual data over QC flags
    if 'PSAL' in ds.variables:
        var_map['sal'] = 'PSAL'
    elif 'psal' in ds.variables:
        var_map['sal'] = 'psal'
    elif 'salinity' in ds.variables:
        var_map['sal'] = 'salinity'
    else:
        sal_candidates = [v for v in ds.variables if ('sal' in v.lower() or 'psal' in v.lower()) and '_qc' not in v.lower()]
        var_map['sal'] = sal_candidates[0] if sal_candidates else None
    
    # Platform/Float ID detection - prioritize standard names
    if 'PLATFORM_NUMBER' in ds.variables:
        var_map['platform'] = 'PLATFORM_NUMBER'
    else:
        platform_candidates = [v for v in ds.variables if any(pattern in v.lower() for pattern in ['platform', 'float', 'wmo'])]
        var_map['platform'] = platform_candidates[0] if platform_candidates else None
    
    # Time detection - prioritize standard names
    if 'JULD' in ds.variables:
        var_map['time'] = 'JULD'
    elif 'juld' in ds.variables:
        var_map['time'] = 'juld'
    else:
        time_candidates = [v for v in ds.variables if any(pattern in v.lower() for pattern in ['juld', 'time', 'date']) and '_qc' not in v.lower()]
        var_map['time'] = time_candidates[0] if time_candidates else None
    
    # Cycle number detection
    if 'CYCLE_NUMBER' in ds.variables:
        var_map['cycle'] = 'CYCLE_NUMBER'
    elif 'cycle_number' in ds.variables:
        var_map['cycle'] = 'cycle_number'
    else:
        cycle_candidates = [v for v in ds.variables if 'cycle' in v.lower()]
        var_map['cycle'] = cycle_candidates[0] if cycle_candidates else None
    
    # Cycle number detection - prioritize standard names
    if 'CYCLE_NUMBER' in ds.variables:
        var_map['cycle'] = 'CYCLE_NUMBER'
    else:
        cycle_candidates = [v for v in ds.variables if any(pattern in v.lower() for pattern in ['cycle', 'profile'])]
        var_map['cycle'] = cycle_candidates[0] if cycle_candidates else None
    
    return var_map

def extract_platform_number(ds, var_map):
    """Extract platform number with robust handling"""
    if var_map['platform'] is None:
        return "UNKNOWN"
    
    try:
        platform_data = ds[var_map['platform']].values
        
        # Handle different data types
        if platform_data.ndim > 0:
            platform_val = platform_data.flat[0]
        else:
            platform_val = platform_data
            
        # Handle bytes, strings, numbers
        if isinstance(platform_val, bytes):
            return platform_val.decode('utf-8').strip()
        elif isinstance(platform_val, str):
            return platform_val.strip()
        else:
            return str(platform_val).strip()
            
    except Exception as e:
        print(f"Warning: Could not extract platform number: {e}")
        return "UNKNOWN"

def process_argo_robust(netcdf_file):
    """Robust ARGO file processor that handles various formats"""
    print(f"\n🔍 Processing: {Path(netcdf_file).name}")
    
    try:
        with xr.open_dataset(netcdf_file) as ds:
            # Smart variable detection
            var_map = smart_variable_detection(ds)
            print(f"Detected variables: {var_map}")
            
            # Check if we have minimum required variables
            required_vars = ['lat', 'lon', 'pres', 'temp']
            missing_vars = [var for var in required_vars if var_map[var] is None]
            
            if missing_vars:
                print(f"❌ Missing required variables: {missing_vars}")
                return None
            
            # Get platform number
            platform_number = extract_platform_number(ds, var_map)
            print(f"Platform number: {platform_number}")
            
            # Determine dimensions
            lat_dims = ds[var_map['lat']].dims
            temp_dims = ds[var_map['temp']].dims
            
            print(f"Latitude dimensions: {lat_dims}, shape: {ds[var_map['lat']].shape}")
            print(f"Temperature dimensions: {temp_dims}, shape: {ds[var_map['temp']].shape}")
            
            all_records = []
            
            # Case 1: Profile-based data (most common ARGO format)
            if len(lat_dims) > 0 and len(temp_dims) > 1:
                n_profiles = ds[var_map['lat']].shape[0]
                n_levels = ds[var_map['temp']].shape[-1]
                
                print(f"Profile-based data: {n_profiles} profiles, {n_levels} levels each")
                
                for i in range(n_profiles):
                    try:
                        # Get profile location and time
                        lat_val = float(ds[var_map['lat']][i].values)
                        lon_val = float(ds[var_map['lon']][i].values)
                        
                        # Get time if available
                        if var_map['time']:
                            time_val = convert_julian_date(ds[var_map['time']][i].values)
                        else:
                            time_val = "2024-01-01"  # Default
                        
                        # Get cycle number if available
                        if var_map['cycle']:
                            cycle_val = int(ds[var_map['cycle']][i].values)
                        else:
                            cycle_val = i + 1
                        
                        # Get all measurements for this profile
                        temp_profile = ds[var_map['temp']][i].values.astype(float)
                        pres_profile = ds[var_map['pres']][i].values.astype(float)
                        
                        if var_map['sal']:
                            sal_profile = ds[var_map['sal']][i].values.astype(float)
                        else:
                            sal_profile = np.full(len(temp_profile), np.nan)
                        
                        # Create records for each depth level
                        valid_indices = ~(np.isnan(temp_profile.astype(float)) | np.isnan(pres_profile.astype(float)))
                        
                        if valid_indices.sum() > 0:
                            profile_data = pd.DataFrame({
                                'platform_number': platform_number,
                                'latitude': lat_val,
                                'longitude': lon_val,
                                'date': time_val,
                                'cycle_number': cycle_val,
                                'pressure': pres_profile[valid_indices],
                                'temperature': temp_profile[valid_indices],
                                'salinity': sal_profile[valid_indices]
                            })
                            all_records.append(profile_data)
                            
                    except Exception as e:
                        print(f"Error processing profile {i}: {e}")
                        continue
            
            # Case 2: Single location, multiple measurements
            elif len(lat_dims) == 0 or ds[var_map['lat']].shape[0] == 1:
                print("Single location data")
                
                lat_val = float(ds[var_map['lat']].values.flat[0])
                lon_val = float(ds[var_map['lon']].values.flat[0])
                
                temp_data = ds[var_map['temp']].values.flatten().astype(float)
                pres_data = ds[var_map['pres']].values.flatten().astype(float)
                
                if var_map['sal']:
                    sal_data = ds[var_map['sal']].values.flatten().astype(float)
                else:
                    sal_data = np.full(len(temp_data), np.nan)
                
                valid_indices = ~(np.isnan(temp_data) | np.isnan(pres_data))
                
                if valid_indices.sum() > 0:
                    profile_data = pd.DataFrame({
                        'platform_number': platform_number,
                        'latitude': lat_val,
                        'longitude': lon_val,
                        'date': "2024-01-01",
                        'cycle_number': 1,
                        'pressure': pres_data[valid_indices],
                        'temperature': temp_data[valid_indices],
                        'salinity': sal_data[valid_indices]
                    })
                    all_records.append(profile_data)
            
            # Combine all records
            if all_records:
                combined_df = pd.concat(all_records, ignore_index=True)
                print(f"✅ Extracted {len(combined_df)} valid measurements")
                return combined_df
            else:
                print("❌ No valid records found")
                return None
                
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_processor():
    """Test the robust processor on your files"""
    # Look for NetCDF files in multiple locations
    nc_files = []
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        nc_files.extend(list(data_dir.glob("*.nc")))
    
    # Check current directory for additional NetCDF files
    current_dir = Path(".")
    nc_files.extend([f for f in current_dir.glob("*.nc") if f.name not in [ncf.name for ncf in nc_files]])
    
    print(f"🧪 Testing robust processor on {len(nc_files)} files...")
    print(f"Found NetCDF files: {[f.name for f in nc_files]}")
    
    all_data = []
    for nc_file in nc_files:
        df = process_argo_robust(nc_file)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n🎉 SUCCESS! Total extracted: {len(combined)} measurements")
        print(f"Temperature range: {combined['temperature'].min():.2f}°C to {combined['temperature'].max():.2f}°C")
        print(f"Platforms: {combined['platform_number'].unique()}")
        print(f"Geographic range: {combined['latitude'].min():.2f}°N to {combined['latitude'].max():.2f}°N")
        
        # Save processed data
        combined.to_csv("processed_argo_data.csv", index=False)
        print("💾 Saved processed data to 'processed_argo_data.csv'")
        
        return combined
    else:
        print("❌ No data could be extracted from any file")
        return None

if __name__ == "__main__":
    test_processor()