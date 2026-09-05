"""
MCP Client that connects to the Google Tools MCP Server
and exposes the tools as LangChain-compatible tools.
"""

import json
import subprocess
import sys
from langchain_core.tools import tool


def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Calls a tool on the MCP server via subprocess.
    In production this would use the MCP client SDK over stdio.
    """
    try:
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server/google_tools_server.py"],
        )

        async def run():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    if result.content:
                        return result.content[0].text
                    return "{}"

        return asyncio.run(run())
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def knowledge_graph_search(company_name: str) -> str:
    """
    Enriches a Colombian company profile using Google Knowledge Graph API.
    Returns sector, description, website and metadata for a given company name.
    """
    return _call_mcp_tool("knowledge_graph_search", {"company_name": company_name})


@tool
def custom_search(query: str, num_results: int = 5) -> str:
    """
    Searches for recent news about a company or sector using Google Custom Search API.
    Focused on expansion, IT investment, mergers, leadership changes and new projects.
    """
    return _call_mcp_tool("custom_search", {"query": query, "num_results": num_results})


@tool
def news_rss(sector: str, city: str, max_items: int = 5) -> str:
    """
    Monitors Google News RSS feed for a given sector and city in Colombia.
    Returns recent news articles relevant to commercial intelligence.
    """
    return _call_mcp_tool("news_rss", {"sector": sector, "city": city, "max_items": max_items})