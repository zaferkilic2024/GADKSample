from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio


from dotenv import load_dotenv
load_dotenv()

tools = MCPToolset(
    connection_params=StdioServerParameters(
        command='npx',
        args=[
            "-y",  # Argument for npx to auto-confirm install
            "@modelcontextprotocol/server-filesystem",
            "C:\Zafer\Kod\Agents\SimpleAgent\Temp",
        ],
    ),
)

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='filesystem_assistant_agent',
    instruction='Help the user manage their files. You can list files, read files, etc.',
    tools=[tools]
)

from event_utils import handle_event_response
APP_NAME = "test_app"
USER_ID = "1234"
SESSION_ID = "session1234"

async def call_agent(query):
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print(f"\nUser Query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])

    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    async for event in events:
        handle_event_response(event)

    await tools.close()
    await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(call_agent("C:\Zafer\Kod\Agents\SimpleAgent\Temp dizinindeki dosyaları listele"))
