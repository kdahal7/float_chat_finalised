#!/usr/bin/env python3
"""
ARGO FloatChat MCP Integration

Model Context Protocol integration for ARGO ocean data analysis.
This provides enhanced AI capabilities for oceanographic queries.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mimeType: str

@dataclass
class MCPTool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ArgoMCPServer:
    def __init__(self, db_path: str = "floatchat.db"):
        self.db_path = db_path
        self.resources = []
        self.tools = []
        self.setup_resources_and_tools()
    
    def setup_resources_and_tools(self):
        """Setup MCP resources and tools for ARGO data"""
        
        # Define MCP Resources
        self.resources = [
            MCPResource(
                uri="argo://database/measurements",
                name="ARGO Measurements Database", 
                description="Complete ARGO float measurements with temperature, salinity, pressure data",
                mimeType="application/json"
            ),
            MCPResource(
                uri="argo://floats/platforms",
                name="ARGO Float Platforms",
                description="Information about ARGO float platforms and their trajectories",
                mimeType="application/json"
            ),
            MCPResource(
                uri="argo://regions/indian_ocean",
                name="Indian Ocean ARGO Data",
                description="ARGO measurements specifically from the Indian Ocean region",
                mimeType="application/json"
            )
        ]
        
        # Define MCP Tools
        self.tools = [
            MCPTool(
                name="query_argo_data",
                description="Execute SQL queries on ARGO float database for oceanographic analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute on ARGO database"
                        },
                        "limit": {
                            "type": "integer", 
                            "description": "Maximum number of results to return",
                            "default": 100
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="get_argo_statistics",
                description="Get statistical analysis of ARGO data (count, averages, ranges)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "parameter": {
                            "type": "string",
                            "enum": ["temperature", "salinity", "pressure", "all"],
                            "description": "Parameter to analyze"
                        },
                        "region": {
                            "type": "string",
                            "description": "Geographic region filter (optional)"
                        }
                    },
                    "required": ["parameter"]
                }
            ),
            MCPTool(
                name="find_nearby_floats",
                description="Find ARGO floats near specified coordinates",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "Latitude coordinate"},
                        "longitude": {"type": "number", "description": "Longitude coordinate"}, 
                        "radius": {"type": "number", "description": "Search radius in degrees", "default": 1.0}
                    },
                    "required": ["latitude", "longitude"]
                }
            )
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return await self.initialize()
        elif method == "resources/list":
            return await self.list_resources()
        elif method == "resources/read":
            return await self.read_resource(params)
        elif method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            return await self.call_tool(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP server"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {"subscribe": True, "listChanged": True},
                "tools": {},
                "prompts": {},
                "logging": {}
            },
            "serverInfo": {
                "name": "argo-mcp-server",
                "version": "1.0.0"
            }
        }
    
    async def list_resources(self) -> Dict[str, Any]:
        """List available MCP resources"""
        return {
            "resources": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                }
                for resource in self.resources
            ]
        }
    
    async def read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read MCP resource content"""
        uri = params.get("uri")
        
        if uri == "argo://database/measurements":
            return await self.get_measurements_summary()
        elif uri == "argo://floats/platforms":
            return await self.get_platforms_info()
        elif uri == "argo://regions/indian_ocean":
            return await self.get_indian_ocean_data()
        else:
            return {"error": f"Resource not found: {uri}"}
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in self.tools
            ]
        }
    
    async def call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "query_argo_data":
            return await self.query_argo_data(arguments)
        elif name == "get_argo_statistics":
            return await self.get_argo_statistics(arguments)
        elif name == "find_nearby_floats":
            return await self.find_nearby_floats(arguments)
        else:
            return {"error": f"Tool not found: {name}"}
    
    async def query_argo_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute natural language query on ARGO database using SQL chain"""
        try:
            query = args.get("query")
            limit = args.get("limit", 100)
            
            # Import SQL chain for natural language processing
            try:
                from sql_chain import text_to_sql_chain, TEXT_TO_SQL_AVAILABLE
                
                if TEXT_TO_SQL_AVAILABLE:
                    # Use the text-to-SQL chain for natural language queries
                    result = text_to_sql_chain.invoke({"question": query})
                    
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Query: {query}\n\nAnswer: {result}"
                            }
                        ]
                    }
                else:
                    # Fallback to direct SQL execution if available
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Try to execute as SQL query
                    if "LIMIT" not in query.upper():
                        query = f"{query} LIMIT {limit}"
                    
                    cursor.execute(query)
                    results = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    conn.close()
                    
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "query": query,
                                    "results": [dict(zip(columns, row)) for row in results],
                                    "count": len(results)
                                }, indent=2)
                            }
                        ]
                    }
            except ImportError:
                # Fallback for direct SQL
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                conn.close()
                
                return {
                    "content": [
                        {
                            "type": "text", 
                            "text": json.dumps({
                                "results": [dict(zip(columns, row)) for row in results],
                                "count": len(results)
                            }, indent=2)
                        }
                    ]
                }
                
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
    
    async def get_measurements_summary(self) -> Dict[str, Any]:
        """Get summary of all ARGO measurements"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_measurements,
                    COUNT(DISTINCT platform_number) as unique_platforms,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    AVG(temperature) as avg_temperature,
                    AVG(salinity) as avg_salinity,
                    AVG(pressure) as avg_pressure
                FROM argo_measurements
            """)
            
            result = cursor.fetchone()
            columns = [description[0] for description in cursor.description]
            summary = dict(zip(columns, result))
            
            conn.close()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(summary, indent=2)
                    }
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get summary: {str(e)}"}

# Global MCP server instance
mcp_server = ArgoMCPServer()

async def process_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process MCP requests"""
    return await mcp_server.handle_request(request)