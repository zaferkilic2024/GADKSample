# adk_mcp_server.py
import asyncio
import json
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types # Use alias to avoid conflict with genai.types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_web_page import load_web_page # Example ADK tool
# ADK <-> MCP Conversion Utility
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# --- Load Environment Variables (If ADK tools need them) ---
load_dotenv()

import os

def append_to_file(directory: str, file_path: str, text: str) -> dict:

    try:
        full_path = directory + "\\" + file_path
        with open(full_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(text + "\n")

        return {
            "status": "success",
            "message": f"Basariyla eklendi: {full_path}",
            "written_text": text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Istisna olustu: {str(e)}",
            "file_path": file_path,
            "text": text
        }

# --- Define Additional ADK Tool: add_numbers ---
def add_numbers(a: int, b: int) -> dict:
    """Toplama işlemi yapan basit bir ADK aracı."""
    return {"result": a + b + 1}

from datetime import datetime

def get_current_datetime(dummy: str) -> dict:
    try:
        dummy = len(dummy)
        current_datetime = datetime.now().strftime("[%Y.%m.%d %H:%M]")
        #current_datetime = "zafer" #str(now.strftime("%Y-%m-%d %H:%M"))
        return {
            "status": "success",
            "message": f"Basarili",
            "current_datetime": current_datetime
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Hata olustu: {str(e)}",
        }
    
print ("datetime = ", get_current_datetime("DUMMY"))
# --- End Additional Tool Definition ---

# --- Prepare the ADK Tools ---
print("Initializing ADK tools...")
adk_web_tool = FunctionTool(load_web_page)
print(f"ADK tool '{adk_web_tool.name}' initialized.")
adk_add_tool = FunctionTool(add_numbers)
print(f"ADK tool '{adk_add_tool.name}' initialized.")
adk_append_tool = FunctionTool(append_to_file)
print(f"ADK tool '{adk_append_tool.name}' initialized.")
adk_datetime_tool = FunctionTool(get_current_datetime)
print(f"ADK tool '{adk_datetime_tool.name}' initialized.")
# --- End ADK Tool Prep ---

# --- MCP Server Setup ---
print("Creating MCP Server instance...")
app = Server("adk-web-tool-mcp-server")

# Implement the MCP server's @app.list_tools handler
@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    """MCP handler to list available tools."""
    print("MCP Server: Received list_tools request.")
    mcp_web_tool_schema = adk_to_mcp_tool_type(adk_web_tool)
    mcp_add_tool_schema = adk_to_mcp_tool_type(adk_add_tool)
    mcp_datetime_tool_schema = adk_to_mcp_tool_type(adk_datetime_tool)
    mcp_append_tool_schema = adk_to_mcp_tool_type(adk_append_tool)
    
    print(f"MCP Server: Advertising tools: {mcp_web_tool_schema.name}, {mcp_add_tool_schema.name}, {mcp_append_tool_schema}, {mcp_datetime_tool_schema}")
    return [mcp_web_tool_schema, mcp_add_tool_schema, mcp_datetime_tool_schema, mcp_append_tool_schema]

# Implement the MCP server's @app.call_tool handler
@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
    """MCP handler to execute a tool call."""
    print(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    try:
        if name == adk_web_tool.name:
            adk_response = await adk_web_tool.run_async(args=arguments, tool_context=None)
        elif name == adk_add_tool.name:
            adk_response = await adk_add_tool.run_async(args=arguments, tool_context=None)
        elif name == adk_datetime_tool.name:
            adk_response = await adk_datetime_tool.run_async(args=arguments, tool_context=None)
        elif name == adk_append_tool.name:
            try:
                adk_response = await adk_append_tool.run_async(args=arguments, tool_context=None)
                print(f"append_to_file response: {adk_response}")
                response_text = json.dumps(adk_response, indent=2)
                return [mcp_types.TextContent(type="text", text=response_text)]
            except Exception as e:
                print(f"append_to_file EXCEPTION: {e}")
                return [mcp_types.TextContent(type="text", text=f"Tool exception: {str(e)}")]
        else:
            raise ValueError(f"Tool '{name}' not implemented.")

        response_text = json.dumps(adk_response, indent=2)
        return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
        print(f"MCP Server: Error executing tool '{name}': {e}")
        error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_server():
    """Runs the MCP server over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("MCP Server starting handshake...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        print("MCP Server run loop finished.")

if __name__ == "__main__":
    print("Launching MCP Server exposing ADK tools...")
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nMCP Server stopped by user.")
    except Exception as e:
        print(f"MCP Server encountered an error: {e}")
    finally:
        print("MCP Server process exiting.")
