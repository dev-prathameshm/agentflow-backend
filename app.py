import os
import json
import chainlit as cl
from fastapi import WebSocket, WebSocketDisconnect
from chainlit.server import app as fastapi_app
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Free Fast LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    api_key=os.environ.get("GROQ_API_KEY") 
)

# =====================================================================
# 1. DEFINE TOOLS (Agentic Capabilities)
# =====================================================================
@tool
def fetch_analytics_data(app_name: str) -> str:
    """Fetches real-time revenue and user data for a given app."""
    # Mock database dip
    return f"Data for {app_name}: 1,250 Active Users, $4,320 MRR. Trend is up 12%."

tools = [fetch_analytics_data]

# =====================================================================
# 2. CONFIGURE AGENT
# =====================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant helping an app developer manage their portfolio. Use tools to fetch data when asked."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# =====================================================================
# 3. CHAINLIT WEB UI LOGIC (Kept for easy browser debugging)
# =====================================================================
@cl.on_message
async def main(message: cl.Message):
    res = await cl.make_async(agent_executor.invoke)(
        {"input": message.content},
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res["output"]).send()

# =====================================================================
# 4. CUSTOM WEBSOCKET FOR FLUTTER (Streams Agent Thoughts & Tokens)
# =====================================================================
@fastapi_app.websocket("/api/chat/ws")
async def flutter_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Flutter client connected to WebSocket!")
    
    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"Received from Flutter: {user_text}")
            
            # Tell Flutter the stream is initiating
            await websocket.send_json({"type": "stream_start"})
            
            try:
                # Use astream_events to peek inside the agent's brain
                async for event in agent_executor.astream_events(
                    {"input": user_text},
                    version="v2"
                ):
                    kind = event["event"]
                    
                    # 1. Catch final text tokens
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if isinstance(content, str) and content:
                            await websocket.send_json({"type": "token", "content": content})
                            
                    # 2. Catch tool start (Agent Thought)
                    elif kind == "on_tool_start":
                        tool_name = event["name"]
                        await websocket.send_json({"type": "thought", "content": f"⚙️ Executing: {tool_name}..."})
                        
                    # 3. Catch tool end (Agent Thought)
                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        await websocket.send_json({"type": "thought", "content": f"✅ {tool_name} finished."})

                # Tell Flutter the message is entirely complete
                await websocket.send_json({"type": "stream_end"})
                
            except Exception as e:
                print(f"Agent error: {e}")
                await websocket.send_json({"type": "error", "content": f"LLM Error: {str(e)}"})
                
    except WebSocketDisconnect:
        print("📱 Flutter client disconnected.")