"""
MCP Server exposing Google tools for the B2B Commercial Intelligence Agent.
Tools exposed:
- knowledge_graph_search: enriches company data via Google Knowledge Graph API
- custom_search: searches expansion and IT investment news via Google Custom Search API
- news_rss: monitors Google News RSS by sector and city
"""

import asyncio
import json
import os
import feedparser
import requests
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

app = Server("google-tools-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="knowledge_graph_search",
            description=(
                "Enriches a Colombian company profile using Google Knowledge Graph API. "
                "Returns sector, description, website and metadata for a given company name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Name of the Colombian company to enrich"
                    }
                },
                "required": ["company_name"]
            }
        ),
        Tool(
            name="custom_search",
            description=(
                "Searches for recent news about a company or sector using Google Custom Search API. "
                "Focused on expansion, IT investment, mergers, leadership changes and new projects."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query combining company name and signal type"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="news_rss",
            description=(
                "Monitors Google News RSS feed for a given sector and city in Colombia. "
                "Returns recent news articles relevant to commercial intelligence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Business sector to monitor (e.g. manufacturing, retail)"
                    },
                    "city": {
                        "type": "string",
                        "description": "Colombian city to focus the search (e.g. Bogota, Medellin)"
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of news items to return",
                        "default": 5
                    }
                },
                "required": ["sector", "city"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "knowledge_graph_search":
        return await _knowledge_graph_search(arguments)
    elif name == "custom_search":
        return await _custom_search(arguments)
    elif name == "news_rss":
        return await _news_rss(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _knowledge_graph_search(arguments: dict) -> list[TextContent]:
    company_name = arguments.get("company_name", "")
    try:
        url = "https://kgsearch.googleapis.com/v1/entities:search"
        params = {
            "query": company_name,
            "key": GOOGLE_API_KEY,
            "limit": 3,
            "indent": True,
            "types": "Organization"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("itemListElement", []):
            entity = item.get("result", {})
            result = {
                "name": entity.get("name", ""),
                "description": entity.get("description", ""),
                "detailed_description": entity.get("detailedDescription", {}).get("articleBody", ""),
                "url": entity.get("url", ""),
                "score": item.get("resultScore", 0)
            }
            results.append(result)

        if not results:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "company": company_name,
                    "found": False,
                    "message": "No Knowledge Graph results found for this company"
                })
            )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "company": company_name,
                "found": True,
                "results": results
            }, ensure_ascii=False)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "company": company_name,
                "found": False,
                "error": str(e)
            })
        )]


async def _custom_search(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 5)
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": min(num_results, 10),
            "lr": "lang_es",
            "gl": "co"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            result = {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": item.get("displayLink", "")
            }
            results.append(result)

        return [TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "total_results": len(results),
                "results": results
            }, ensure_ascii=False)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "total_results": 0,
                "error": str(e)
            })
        )]


async def _news_rss(arguments: dict) -> list[TextContent]:
    sector = arguments.get("sector", "")
    city = arguments.get("city", "")
    max_items = arguments.get("max_items", 5)
    try:
        query = f"{sector} empresas {city} Colombia expansion inversion tecnologia"
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=es-419&gl=CO&ceid=CO:es-419"

        feed = feedparser.parse(rss_url)
        items = []
        for entry in feed.entries[:max_items]:
            item = {
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "url": entry.get("link", "")
            }
            items.append(item)

        return [TextContent(
            type="text",
            text=json.dumps({
                "sector": sector,
                "city": city,
                "items_found": len(items),
                "news": items
            }, ensure_ascii=False)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "sector": sector,
                "city": city,
                "items_found": 0,
                "error": str(e)
            })
        )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())