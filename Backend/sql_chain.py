from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
import re

load_dotenv()

class QueryRequest(BaseModel):
    question: str

router = APIRouter()

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///floatchat.db")

try:
    # --- Database and LLM Setup ---
    db = SQLDatabase.from_uri(DATABASE_URL)
    
    # Configure database to show more representative sample data
    db._sample_rows_in_table_info = 10
    
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
        num_predict=8192,  # Ollama parameter for max output tokens
        num_ctx=4096,      # Context window
        top_p=0.9
    )

    # --- Function to extract SQL query from formatted output ---
    def extract_sql_query(query_output):
        """Extract SQL query from the formatted output and fix common syntax issues"""
        print(f"🔍 DEBUG - Raw LLM output: {query_output}")
        
        if isinstance(query_output, str):
            # Remove any markdown formatting
            query_output = query_output.replace('```sql', '').replace('```', '')
            
            # Look for SQLQuery: pattern first
            match = re.search(r'SQLQuery:\s*(.*?)(?:\n|$)', query_output, re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                sql = fix_sql_syntax(sql)
                print(f"🔍 DEBUG - Extracted SQL (method 1): {sql}")
                return sql.rstrip(';')  # Remove trailing semicolon
            
            # Look for SQL Query: pattern
            match = re.search(r'SQL Query:\s*(.*?)(?:\n|$)', query_output, re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                sql = fix_sql_syntax(sql)
                print(f"🔍 DEBUG - Extracted SQL (method 2): {sql}")
                return sql.rstrip(';')
            
            # Fallback: try to find SELECT statement (multi-line)
            match = re.search(r'(SELECT.*?)(?:\n\n|$)', query_output, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                sql = fix_sql_syntax(sql)
                print(f"🔍 DEBUG - Extracted SQL (method 3): {sql}")
                return sql.rstrip(';')
            
            # Last resort: clean the entire output
            cleaned = query_output.strip()
            if cleaned.upper().startswith('SELECT'):
                sql = fix_sql_syntax(cleaned)
                print(f"🔍 DEBUG - Extracted SQL (method 4): {sql}")
                return sql.rstrip(';')
                
        final_sql = fix_sql_syntax(str(query_output).strip())
        print(f"🔍 DEBUG - Final SQL: {final_sql}")
        return final_sql.rstrip(';')

    def fix_sql_syntax(sql_query):
        """Fix common SQL syntax issues for SQLite"""
        if not sql_query:
            return sql_query
            
        # Fix PostgreSQL-style EXTRACT functions for SQLite
        sql_query = re.sub(r'EXTRACT\(YEAR FROM ([^)]+)\)', r'strftime("%Y", \1)', sql_query)
        sql_query = re.sub(r'EXTRACT\(MONTH FROM ([^)]+)\)', r'strftime("%m", \1)', sql_query)
        sql_query = re.sub(r'EXTRACT\(DAY FROM ([^)]+)\)', r'strftime("%d", \1)', sql_query)
        
        # Fix LIMIT clauses that might have syntax issues
        sql_query = re.sub(r'LIMIT\s+(\d+)\s*;?\s*$', r'LIMIT \1', sql_query, flags=re.IGNORECASE)
        
        return sql_query

    # --- Chain for Generating SQL Query ---
    # Use the default chain but with better database configuration
    generate_query_chain = create_sql_query_chain(llm, db)

    # --- Tool for Executing the SQL Query ---
    def debug_sql_execution(sql_query):
        """Debug wrapper for SQL execution"""
        print(f"🔍 DEBUG - Executing SQL: {sql_query}")
        try:
            result = execute_query_tool.run(sql_query)
            print(f"🔍 DEBUG - SQL Result: {str(result)[:200]}...")
            return result
        except Exception as e:
            print(f"🔍 DEBUG - SQL Error: {e}")
            raise e
    
    # --- Function to get dynamic database context ---
    def get_database_context():
        """Get current database context dynamically"""
        try:
            # Get basic stats using execute_query_tool
            total_result = execute_query_tool.run("SELECT COUNT(*) FROM argo_measurements")
            # Parse the result - it comes as a string like "[(7762,)]"
            import re
            total_match = re.search(r'\((\d+),?\)', str(total_result))
            total_measurements = int(total_match.group(1)) if total_match else 0
            
            # Get platform list
            platforms_result = execute_query_tool.run("SELECT DISTINCT platform_number FROM argo_measurements ORDER BY platform_number LIMIT 10")
            platform_matches = re.findall(r'\(\'?(\d+)\'?,?\)', str(platforms_result))
            platforms = platform_matches[:10] if platform_matches else []
            
            # Get temperature range
            temp_result = execute_query_tool.run("SELECT MIN(temperature), MAX(temperature) FROM argo_measurements WHERE temperature IS NOT NULL")
            temp_match = re.search(r'\(([^,]+),\s*([^)]+)\)', str(temp_result))
            if temp_match:
                min_temp = temp_match.group(1).strip()
                max_temp = temp_match.group(2).strip()
            else:
                min_temp = max_temp = "Unknown"
            
            return f"""CRITICAL DATABASE CONTEXT:
- Our database contains {total_measurements:,} ARGO measurements from Indian Ocean
- Temperature range: {min_temp}°C to {max_temp}°C  
- {len(platforms)} ARGO float platforms: {', '.join(platforms)}
- Indian Ocean focus with comprehensive depth profiles
- pressure column = ocean depth (decibars ≈ meters)"""
        except Exception as e:
            print(f"⚠️ Error getting database context: {e}")
            return """CRITICAL DATABASE CONTEXT:
- Our database contains ARGO measurements from Indian Ocean
- Indian Ocean focus with comprehensive depth profiles
- pressure column = ocean depth (decibars ≈ meters)"""
    
    execute_query_tool = QuerySQLDataBaseTool(db=db)

    # --- Prompt for Generating the Final Natural Language Answer ---
    answer_prompt = PromptTemplate.from_template(
        """You are FloatChat, an AI-powered oceanographic data analyst specializing in ARGO float data from the Indian Ocean.

{database_context}

Based on the SQL query results, provide a comprehensive, professional answer:

1. **Direct Answer**: Answer with exact numbers from the SQL results
2. **Data Details**: Include specific measurements, coordinates, platform numbers, dates
3. **Oceanographic Context**: Explain significance of patterns when relevant
4. **Geographic Focus**: Reference Indian Ocean locations and conditions

IMPORTANT: 
- Use ONLY the actual SQL results provided. Do not invent or hallucinate data!
- If SQL results are empty or show no data found, clearly state "No data found" 
- If a platform/measurement doesn't exist in our database, acknowledge this fact
- Never fabricate coordinates, measurements, or dates that aren't in the SQL results

User Question: {question}
SQL Query: {query}
SQL Results: {result}

Professional Oceanographic Analysis:"""
    )

    # --- Dedicated LLM for longer responses ---
    rephrase_llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.1,
        num_predict=16384,  # Very high limit for complete responses
        num_ctx=8192,       # Larger context window
        top_p=0.9,
        repeat_penalty=1.1
    )
    
    # --- Chain for Rephrasing the SQL Result into a Natural Language Answer ---
    rephrase_answer_chain = answer_prompt | rephrase_llm | StrOutputParser()

    # --- The Full Text-to-SQL RAG Chain ---
    text_to_sql_chain = (
        RunnablePassthrough.assign(query=generate_query_chain).assign(
            result=itemgetter("query") | RunnableLambda(extract_sql_query) | RunnableLambda(debug_sql_execution)
        ).assign(
            database_context=RunnableLambda(lambda _: get_database_context())
        )
        | rephrase_answer_chain
    )
    TEXT_TO_SQL_AVAILABLE = True
    print("✅ Text-to-SQL pipeline loaded successfully with Ollama!")

except Exception as e:
    TEXT_TO_SQL_AVAILABLE = False
    print(f"⚠️ Text-to-SQL pipeline failed to load: {e}")


@router.post("/sql_query")
def sql_query(request: QueryRequest):
    if not TEXT_TO_SQL_AVAILABLE:
        raise HTTPException(status_code=500, detail="Text-to-SQL pipeline is not available. Check backend logs.")

    try:
        print(f"Processing SQL query: {request.question}")
        result = text_to_sql_chain.invoke({"question": request.question})
        print(f"SQL query result: {result[:200]}...")  # Log first 200 chars
        return {"answer": result}
    except Exception as e:
        print(f"Text-to-SQL query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process query. Error: {e}")


def get_enhanced_sql_chain():
    """Return the text-to-SQL chain for direct use"""
    if TEXT_TO_SQL_AVAILABLE:
        return text_to_sql_chain
    else:
        print("⚠️ SQL chain is not available - returning None instead of raising exception")
        return None