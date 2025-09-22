import os
import time
import hashlib
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # Remove Field import since we're not using it
import uvicorn
from dotenv import load_dotenv
import logging
from sqlalchemy import text  # Import text for raw SQL queries

# Import existing modules
from database import get_db_session, ArgoMeasurement
from sql_chain import get_enhanced_sql_chain
from semantic_chain import get_semantic_chain
from argo_mcp_server import process_mcp_request, mcp_server

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Initializing FloatChat Complete RAG System...")

# Initialize FastAPI app
app = FastAPI(title="FloatChat API", description="AI-Powered ARGO Float Data Explorer", version="2.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for chains and cache
sql_chain = None
semantic_chain = None
response_cache: Dict[str, Any] = {}
CACHE_SIZE_LIMIT = 100

# Dynamic database stats
def get_database_stats():
    """Get current database statistics dynamically"""
    try:
        from database import SessionLocal
        session = SessionLocal()
        
        # Get total measurements
        total_query = session.execute(text("SELECT COUNT(*) FROM argo_measurements")).fetchone()
        total_measurements = total_query[0] if total_query else 0
        
        # Get unique platforms
        platforms_query = session.execute(text("SELECT COUNT(DISTINCT platform_number) FROM argo_measurements")).fetchone()
        unique_platforms = platforms_query[0] if platforms_query else 0
        
        # Get temperature range
        temp_query = session.execute(text("SELECT MIN(temperature), MAX(temperature) FROM argo_measurements WHERE temperature IS NOT NULL")).fetchone()
        min_temp, max_temp = temp_query if temp_query else (None, None)
        
        session.close()
        
        return {
            "total_measurements": total_measurements,
            "unique_platforms": unique_platforms,
            "min_temperature": min_temp,
            "max_temperature": max_temp,
            "formatted_count": f"{total_measurements:,}" if total_measurements > 0 else "0"
        }
    except Exception as e:
        print(f"⚠️ Error getting database stats: {e}")
        return {
            "total_measurements": 0,
            "unique_platforms": 0,
            "min_temperature": None,
            "max_temperature": None,
            "formatted_count": "0"
        }

# Response models (simplified without Field)
class QueryRequest(BaseModel):
    question: str

class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = {}

# Initialize systems
def initialize_systems():
    """Initialize all systems with error handling"""
    global sql_chain, semantic_chain
    
    try:
        # Initialize LLM
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,
            num_ctx=4096,
            num_predict=4096,  # Increased for complete responses
            top_p=0.9
            # Removed stop tokens that were causing truncation
        )
        print("✅ LLM (Ollama) initialized successfully")
        
        # Initialize SQL chain
        sql_chain = get_enhanced_sql_chain()
        if sql_chain:
            print("✅ SQL system initialized")
        else:
            print("⚠️ SQL system failed - continuing with other systems")
            sql_chain = None
        
        # Initialize Semantic chain
        try:
            semantic_chain = get_semantic_chain()
            print("✅ Vector system initialized")
        except Exception as e:
            print(f"⚠️ Vector system failed: {str(e)} - SQL-only mode enabled")
            semantic_chain = None
        
        # Initialize MCP server
        print("✅ MCP system initialized")
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

# Smart query routing
def is_oceanographic_query(question: str) -> bool:
    """Determine if question is related to oceanographic/ARGO data"""
    oceanographic_keywords = [
        'argo', 'float', 'temperature', 'salinity', 'pressure', 'depth', 
        'ocean', 'sea', 'water', 'measurement', 'data', 'platform',
        'profile', 'latitude', 'longitude', 'coordinates', 'indian ocean',
        'arabian sea', 'pacific', 'atlantic', 'celsius', 'psu', 'dbar',
        'oceanographic', 'marine', 'hydrographic', 'ctd', 'bgc'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in oceanographic_keywords)

def get_cache_key(question: str, query_type: str) -> str:
    """Generate cache key for question"""
    return hashlib.md5(f"{question}_{query_type}".encode()).hexdigest()

def cache_response(key: str, response: Any):
    """Cache response with size limit"""
    global response_cache
    if len(response_cache) >= CACHE_SIZE_LIMIT:
        # Remove oldest entry
        oldest_key = next(iter(response_cache))
        del response_cache[oldest_key]
    response_cache[key] = response

def get_casual_response(question: str) -> str:
    """Handle casual/non-oceanographic queries with FloatChat branding"""
    question_lower = question.lower()
    
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in question_lower for greeting in greetings):
        return """🌊 Hello! I'm **FloatChat** - your AI-powered conversational interface for ARGO Ocean Data Discovery and Visualization.

I specialize in Indian Ocean ARGO float data and can help you:
• Query temperature, salinity, and pressure measurements
• Explore ARGO float trajectories and locations  
• Analyze oceanographic patterns and water masses
• Compare data across different regions and time periods
• Visualize ocean profiles and measurements

Try asking: "Show me temperature profiles near the equator" or "What ARGO floats are active in the Indian Ocean?"
"""
    
    if 'how are you' in question_lower:
        stats = get_database_stats()
        return f"""🚀 I'm running perfectly and ready to help you explore ocean data! 

Current system status:
✅ **{stats['formatted_count']} ARGO measurements** from Indian Ocean
✅ **{stats['unique_platforms']} ARGO platforms** actively monitored
✅ **SQL database** for structured queries  
✅ **Vector search** for semantic analysis
✅ **MCP integration** for advanced analytics

What oceanographic data would you like to explore?"""
    
    if 'who are you' in question_lower or 'what are you' in question_lower:
        return """🌊 I'm **FloatChat** - an AI-powered conversational interface for ARGO Ocean Data Discovery and Visualization.

**My capabilities:**
• **Natural Language Queries**: Ask questions in plain English
• **SQL Data Analysis**: Direct database queries for precise results  
• **Semantic Search**: Explore oceanographic concepts and patterns
• **Indian Ocean Focus**: Specialized knowledge of regional ocean dynamics
• **Data Visualization**: Interactive maps and profile comparisons
• **ARGO Float Tracking**: Monitor autonomous profiling floats

I process NetCDF format data and provide insights through RAG (Retrieval-Augmented Generation) powered by advanced language models.

Ready to dive into ocean data exploration!"""
    
    # Default response for other queries
    return """🌊 I'm **FloatChat**, your AI assistant for ARGO oceanographic data from the Indian Ocean. 

For the best results, try queries like:
• "What's the average temperature at 500m depth?"
• "Show me salinity profiles near the equator"  
• "Find ARGO floats in the Arabian Sea"
• "Compare temperature data from different months"
• "What are the water mass characteristics?"

What ocean data would you like to explore?"""

# API Routes
@app.get("/")
async def get_system_status():
    """Get system status"""
    from database import SessionLocal, get_database_info
    session = SessionLocal()
    try:
        total_measurements = session.query(ArgoMeasurement).count()
        unique_platforms = session.query(ArgoMeasurement.platform_number).distinct().count()
        
        # Get current database info
        db_info = get_database_info()
        
        return {
            "system": "FloatChat - AI-Powered ARGO Data Discovery",
            "version": "2.1.0",
            "status": "operational",
            "description": "Conversational Interface for ARGO Ocean Data Discovery and Visualization",
            "capabilities": {
                "sql_queries": sql_chain is not None,
                "semantic_search": semantic_chain is not None,
                "mcp_integration": True,
                "llm_analysis": True,
                "trajectory_mapping": True,
                "profile_comparison": True,
                "data_export": True,
                "nearby_search": True,
                "indian_ocean_focus": True,
                "natural_language_queries": True,
                "netcdf_processing": True,
                "rag_pipeline": True
            },
            "focus_region": "Indian Ocean",
            "database_info": {
                "type": db_info["database_type"],
                "url_preview": db_info["database_url"][:50] + "..." if len(db_info["database_url"]) > 50 else db_info["database_url"],
                "fallback_available": db_info["fallback_available"],
                "description": f"Enterprise {db_info['database_type']} Database" if db_info["database_type"] == "PostgreSQL" else "SQLite Database"
            },
            "vector_store": "FAISS Vector Store" if semantic_chain else "Not Available",
            "mcp_server": "Active",
            "data_summary": {
                "total_measurements": total_measurements,
                "unique_platforms": unique_platforms,
                "region": "Indian Ocean ARGO Data (NetCDF Format)",
                "data_types": ["Temperature", "Salinity", "Pressure/Depth", "Geographic Coordinates"],
                "processing_pipeline": f"NetCDF → {db_info['database_type']} Database + Vector Store → RAG Analysis"
            },
            "example_queries": [
                "How many ARGO measurements are in our database?",
                "What is the geographic coverage of our ARGO data?",
                "Show me depth distribution of our measurements", 
                "Find measurements from platform 1901049",
                "What are the warmest water temperatures recorded?",
                "Compare shallow vs deep water temperatures",
                "Which ARGO platform has the most measurements?",
                "Explain ARGO float technology"
            ]
        }
    finally:
        session.close()

@app.get("/api/database_status")
async def get_database_status():
    """Get detailed database status information"""
    from database import SessionLocal, get_database_info, USE_POSTGRESQL
    session = SessionLocal()
    try:
        # Get database info
        db_info = get_database_info()
        
        # Get some statistics
        total_measurements = session.query(ArgoMeasurement).count()
        unique_platforms = session.query(ArgoMeasurement.platform_number).distinct().count()
        
        # Get a sample query to verify database is working
        from sqlalchemy import text, func
        
        # Test PostgreSQL-specific vs SQLite-specific functions
        if USE_POSTGRESQL:
            # PostgreSQL version info
            version_result = session.execute(text("SELECT version()")).fetchone()
            db_version = version_result[0][:100] + "..." if version_result else "Unknown"
        else:
            # SQLite version info
            version_result = session.execute(text("SELECT sqlite_version()")).fetchone()
            db_version = f"SQLite {version_result[0]}" if version_result else "Unknown"
        
        # Performance test - measure query time
        import time
        start_time = time.time()
        session.execute(text("SELECT COUNT(*) FROM argo_measurements WHERE temperature IS NOT NULL")).fetchone()
        query_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds
        
        return {
            "database_type": db_info["database_type"],
            "database_url": db_info["database_url"][:60] + "..." if len(db_info["database_url"]) > 60 else db_info["database_url"],
            "use_postgresql": USE_POSTGRESQL,
            "fallback_available": db_info["fallback_available"],
            "version": db_version,
            "statistics": {
                "total_measurements": total_measurements,
                "unique_platforms": unique_platforms,
                "sample_query_time_ms": query_time
            },
            "performance_note": "PostgreSQL optimized for complex analytical queries" if USE_POSTGRESQL else "SQLite fallback mode active",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "database_type": "Unknown",
            "status": "Error"
        }
    finally:
        session.close()

@app.post("/api/sql_query")
async def query_with_sql(request: QueryRequest):
    """Handle SQL-based queries with smart routing"""
    if not sql_chain:
        raise HTTPException(status_code=503, detail="SQL system not available")
    
    start_time = time.time()
    question = request.question.strip()
    
    # Check cache first
    cache_key = get_cache_key(question, "sql")
    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        cached_response["cached"] = True
        return cached_response
    
    try:
        # Route based on query type
        if is_oceanographic_query(question):
            # Use direct question for SQL chain
            result = sql_chain.invoke({"question": question})
            # Handle both string and dict results
            if isinstance(result, dict):
                answer = result.get('answer', str(result))
            else:
                answer = str(result)
            
            # Allow full responses without truncation for better user experience
            # Removed truncation limit to show complete oceanographic analysis
            
        else:
            # Handle casual queries
            answer = get_casual_response(question)
        
        response = {
            "question": question,
            "answer": answer,
            "query_type": "sql",
            "processing_time": round(time.time() - start_time, 2),
            "cached": False
        }
        
        # Add map data for location-based queries
        try:
            from database import SessionLocal
            session = SessionLocal()
            
            # Check if query is location-based and get sample data for map
            location_keywords = ['latitude', 'longitude', 'location', 'coordinates', 'where', 'near', 'region', 'ocean', 'sea', 'platform', 'measurements', 'show', 'find', 'temperature', 'salinity']
            if any(keyword in question.lower() for keyword in location_keywords):
                # Get measurements matching the specific query for map visualization
                # Try to extract platform number from question
                import re
                platform_match = re.search(r'platform\s+(\d+)', question.lower())
                
                if platform_match:
                    platform_number = platform_match.group(1)
                    # Get data for specific platform
                    platform_data = session.query(ArgoMeasurement).filter(
                        ArgoMeasurement.platform_number == platform_number,
                        ArgoMeasurement.latitude.isnot(None),
                        ArgoMeasurement.longitude.isnot(None)
                    ).limit(100).all()
                    
                    map_data = []
                    for m in platform_data:
                        map_data.append({
                            "platform_id": m.platform_number,
                            "latitude": float(m.latitude),
                            "longitude": float(m.longitude),
                            "temperature": float(m.temperature) if m.temperature else None,
                            "salinity": float(m.salinity) if m.salinity else None,
                            "pressure": float(m.pressure) if m.pressure else None,
                            "date": m.measurement_date.isoformat() if m.measurement_date else None
                        })
                else:
                    # Get recent measurements with coordinates for general queries
                    recent_data = session.query(ArgoMeasurement).filter(
                        ArgoMeasurement.latitude.isnot(None),
                        ArgoMeasurement.longitude.isnot(None)
                    ).limit(100).all()
                    
                    map_data = []
                    for m in recent_data:
                        map_data.append({
                            "platform_id": m.platform_number,
                            "latitude": float(m.latitude),
                            "longitude": float(m.longitude),
                            "temperature": float(m.temperature) if m.temperature else None,
                            "salinity": float(m.salinity) if m.salinity else None,
                            "pressure": float(m.pressure) if m.pressure else None,
                            "date": m.measurement_date.isoformat() if m.measurement_date else None
                        })
                
                response["map_data"] = map_data
            
            session.close()
        except Exception as map_error:
            logger.warning(f"Could not add map data: {str(map_error)}")
        
        # Cache the response
        cache_response(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"SQL query error: {str(e)}")
        return {
            "question": question,
            "answer": "I encountered an error processing your query. Please try rephrasing your question about ARGO data.",
            "error": str(e),
            "query_type": "sql"
        }

@app.post("/api/semantic_query")
async def query_with_semantic(request: QueryRequest):
    """Handle semantic search queries with smart routing"""
    if not semantic_chain:
        # Fall back to SQL if semantic search unavailable
        return await query_with_sql(request)
    
    start_time = time.time()
    question = request.question.strip()
    
    # Check cache first
    cache_key = get_cache_key(question, "semantic")
    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        cached_response["cached"] = True
        return cached_response
    
    try:
        # Route based on query type
        if is_oceanographic_query(question):
            # Enhanced semantic search with specific context
            enhanced_question = f"""Search the ARGO oceanographic database for: {question}
            
Please provide specific measurements, platform numbers, coordinates, and exact values from the data.
Focus on real data points rather than general descriptions."""
            
            result = semantic_chain.invoke({"query": enhanced_question})
            # Handle both string and dict results  
            if isinstance(result, dict):
                answer = result.get('answer', result.get('result', str(result)))
            else:
                answer = str(result)
                
            # Allow complete semantic search responses for comprehensive analysis
            # Removed truncation to provide full oceanographic context
        else:
            # Handle casual queries
            answer = get_casual_response(question)
        
        response = {
            "question": question,
            "answer": answer,
            "query_type": "semantic",
            "processing_time": round(time.time() - start_time, 2),
            "cached": False
        }
        
        # Add map data for location-based queries
        try:
            from database import SessionLocal
            session = SessionLocal()
            
            # Check if query is location-based and get sample data for map
            location_keywords = ['latitude', 'longitude', 'location', 'coordinates', 'where', 'near', 'region', 'ocean', 'sea', 'float']
            if any(keyword in question.lower() for keyword in location_keywords):
                # Get recent measurements with coordinates for map visualization
                recent_data = session.query(ArgoMeasurement).filter(
                    ArgoMeasurement.latitude.isnot(None),
                    ArgoMeasurement.longitude.isnot(None)
                ).limit(100).all()
                
                map_data = []
                for m in recent_data:
                    map_data.append({
                        "platform_id": m.platform_number,
                        "latitude": float(m.latitude),
                        "longitude": float(m.longitude),
                        "temperature": float(m.temperature) if m.temperature else None,
                        "salinity": float(m.salinity) if m.salinity else None,
                        "pressure": float(m.pressure) if m.pressure else None,
                        "date": m.measurement_date.isoformat() if m.measurement_date else None
                    })
                
                response["map_data"] = map_data
            
            session.close()
        except Exception as map_error:
            logger.warning(f"Could not add map data: {str(map_error)}")
        
        # Cache the response
        cache_response(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Semantic query error: {str(e)}")
        return {
            "question": question,
            "answer": "I encountered an error with semantic search. Please try a different question about ARGO oceanographic data.",
            "error": str(e),
            "query_type": "semantic"
        }

@app.post("/api/mcp_request")
async def handle_mcp_request(request: MCPRequest):
    """Handle MCP (Model Context Protocol) requests"""
    try:
        start_time = time.time()
        
        response = await process_mcp_request({
            "method": request.method,
            "params": request.params
        })
        
        response["processing_time"] = round(time.time() - start_time, 2)
        return response
        
    except Exception as e:
        logger.error(f"MCP request error: {str(e)}")
        return {"error": f"MCP request failed: {str(e)}"}

@app.get("/api/floats")
async def get_floats():
    """Get all ARGO floats with their latest data"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        from sqlalchemy import func
        
        # Get latest measurement for each platform
        subquery = session.query(
            ArgoMeasurement.platform_number,
            func.max(ArgoMeasurement.measurement_date).label('latest_date')
        ).group_by(ArgoMeasurement.platform_number).subquery()
        
        latest_measurements = session.query(ArgoMeasurement).join(
            subquery,
            (ArgoMeasurement.platform_number == subquery.c.platform_number) &
            (ArgoMeasurement.measurement_date == subquery.c.latest_date)
        ).all()
        
        floats = []
        for measurement in latest_measurements:
            if measurement.latitude and measurement.longitude:
                float_data = {
                    "platform_number": measurement.platform_number,
                    "latitude": float(measurement.latitude),
                    "longitude": float(measurement.longitude),
                    "latest_temp": float(measurement.temperature) if measurement.temperature else None,
                    "latest_salinity": float(measurement.salinity) if measurement.salinity else None,
                    "latest_pressure": float(measurement.pressure) if measurement.pressure else None,
                    "latest_date": measurement.measurement_date.isoformat() if measurement.measurement_date else None
                }
                
                # Get measurement count for this platform
                measurement_count = session.query(ArgoMeasurement).filter(
                    ArgoMeasurement.platform_number == measurement.platform_number
                ).count()
                float_data["measurement_count"] = measurement_count
                
                floats.append(float_data)
        
        return {"floats": floats, "total": len(floats)}
        
    finally:
        session.close()

@app.get("/api/trajectory/{platform_number}")
async def get_trajectory(platform_number: str):
    """Get trajectory data for a specific ARGO float"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        measurements = session.query(ArgoMeasurement).filter(
            ArgoMeasurement.platform_number == platform_number
        ).order_by(ArgoMeasurement.measurement_date).all()
        
        if not measurements:
            raise HTTPException(status_code=404, detail="Platform not found")
        
        trajectory = [
            {
                "date": m.measurement_date.isoformat() if m.measurement_date else None,
                "latitude": float(m.latitude) if m.latitude else None,
                "longitude": float(m.longitude) if m.longitude else None,
                "temperature": float(m.temperature) if m.temperature else None,
                "salinity": float(m.salinity) if m.salinity else None,
                "pressure": float(m.pressure) if m.pressure else None
            }
            for m in measurements
        ]
        
        return {
            "platform_number": platform_number,
            "trajectory_points": len(trajectory),
            "trajectory": trajectory
        }
        
    finally:
        session.close()

@app.get("/api/map_data")
async def get_map_data():
    """Get platform data for map visualization"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        from sqlalchemy import func
        
        # Get platform locations with measurement counts
        platform_stats = session.query(
            ArgoMeasurement.platform_number,
            ArgoMeasurement.latitude,
            ArgoMeasurement.longitude,
            func.count(ArgoMeasurement.id).label('measurement_count')
        ).filter(
            ArgoMeasurement.latitude.isnot(None),
            ArgoMeasurement.longitude.isnot(None)
        ).group_by(
            ArgoMeasurement.platform_number,
            ArgoMeasurement.latitude,
            ArgoMeasurement.longitude
        ).all()
        
        static_platforms = []
        for stat in platform_stats:
            static_platforms.append({
                "platform_number": stat.platform_number,
                "latitude": float(stat.latitude),
                "longitude": float(stat.longitude),
                "measurement_count": stat.measurement_count
            })
        
        return {
            "success": True,
            "static_platforms": static_platforms,
            "total_platforms": len(static_platforms)
        }
        
    except Exception as e:
        logger.error(f"Map data error: {str(e)}")
        return {
            "success": False,
            "static_platforms": [],
            "total_platforms": 0,
            "error": str(e)
        }
    finally:
        session.close()

# Initialize systems on startup
initialize_systems()

@app.get("/api/depth_distribution")
async def get_depth_distribution():
    """Get depth distribution data for histogram"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        # Define depth categories and get counts
        depth_query = text("""
        SELECT 
            CASE 
                WHEN pressure < 50 THEN 'Surface (0-50m)'
                WHEN pressure < 200 THEN 'Shallow (50-200m)'
                WHEN pressure < 500 THEN 'Intermediate (200-500m)'
                WHEN pressure < 1000 THEN 'Deep (500-1000m)'
                ELSE 'Very Deep (>1000m)'
            END as depth_category,
            COUNT(*) as count,
            AVG(pressure) as avg_pressure,
            AVG(temperature) as avg_temp
        FROM argo_measurements 
        WHERE pressure IS NOT NULL 
        GROUP BY 
            CASE 
                WHEN pressure < 50 THEN 'Surface (0-50m)'
                WHEN pressure < 200 THEN 'Shallow (50-200m)'
                WHEN pressure < 500 THEN 'Intermediate (200-500m)'
                WHEN pressure < 1000 THEN 'Deep (500-1000m)'
                ELSE 'Very Deep (>1000m)'
            END
        ORDER BY MIN(pressure)
        """)
        
        result = session.execute(depth_query).fetchall()
        
        categories = []
        counts = []
        avg_pressures = []
        avg_temps = []
        
        for row in result:
            categories.append(row[0])
            counts.append(row[1])
            avg_pressures.append(float(row[2]) if row[2] else 0)
            avg_temps.append(float(row[3]) if row[3] else 0)
        
        return {
            "success": True,
            "categories": categories,
            "counts": counts,
            "avg_pressures": avg_pressures,
            "avg_temperatures": avg_temps,
            "total_measurements": sum(counts)
        }
        
    except Exception as e:
        logger.error(f"Depth distribution error: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        session.close()

@app.get("/api/temperature_patterns")
async def get_temperature_patterns():
    """Get temperature vs depth patterns"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        # Get temperature patterns by depth ranges
        temp_query = text("""
        SELECT 
            pressure,
            temperature,
            salinity,
            platform_number
        FROM argo_measurements 
        WHERE temperature IS NOT NULL 
            AND pressure IS NOT NULL
            AND pressure BETWEEN 0 AND 2000
        ORDER BY pressure
        LIMIT 500
        """)
        
        result = session.execute(temp_query).fetchall()
        
        pressures = []
        temperatures = []
        salinities = []
        platforms = []
        
        for row in result:
            pressures.append(float(row[0]))
            temperatures.append(float(row[1]))
            salinities.append(float(row[2]) if row[2] else None)
            platforms.append(row[3])
        
        return {
            "success": True,
            "pressures": pressures,
            "temperatures": temperatures,
            "salinities": salinities,
            "platforms": platforms,
            "count": len(pressures)
        }
        
    except Exception as e:
        logger.error(f"Temperature patterns error: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        session.close()

@app.get("/api/monthly_trends")
async def get_monthly_trends():
    """Get monthly measurement trends"""
    from database import SessionLocal
    session = SessionLocal()
    try:
        # Since measurement_date might be NULL, we'll use platform stats instead
        monthly_query = text("""
        SELECT 
            platform_number,
            COUNT(*) as measurements,
            AVG(temperature) as avg_temp,
            AVG(salinity) as avg_salinity,
            AVG(pressure) as avg_pressure,
            MIN(pressure) as min_depth,
            MAX(pressure) as max_depth
        FROM argo_measurements 
        WHERE temperature IS NOT NULL
        GROUP BY platform_number
        ORDER BY measurements DESC
        LIMIT 10
        """)
        
        result = session.execute(monthly_query).fetchall()
        
        platforms = []
        measurements = []
        avg_temps = []
        avg_salinities = []
        
        for row in result:
            platforms.append(row[0])
            measurements.append(row[1])
            avg_temps.append(float(row[2]) if row[2] else 0)
            avg_salinities.append(float(row[3]) if row[3] else 0)
        
        return {
            "success": True,
            "platforms": platforms,
            "measurements": measurements,
            "avg_temperatures": avg_temps,
            "avg_salinities": avg_salinities,
            "total_platforms": len(platforms)
        }
        
    except Exception as e:
        logger.error(f"Monthly trends error: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        session.close()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)