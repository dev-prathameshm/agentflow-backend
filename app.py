import os
import json
import chainlit as cl
from fastapi import WebSocket, WebSocketDisconnect
from chainlit.server import app as fastapi_app
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Free Fast LLM 
# (Note: Keep your hardcoded API key here if you haven't fixed the .env file yet)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",  # type: ignore
    api_key=os.environ.get("GROQ_API_KEY") 
) # type: ignore

# =====================================================================
# 1. CHAINLIT WEB UI LOGIC (Kept for easy browser debugging)
# =====================================================================
@cl.step(name="processing_request", type="tool")
async def process_intent():
    cl.context.current_step.input = "Analyzing user text..." # type: ignore
    cl.context.current_step.output = "Intent understood." # type: ignore
    return "Success"

@cl.on_message
async def main(message: cl.Message):
    await process_intent()
    messages = [
        SystemMessage(content="You are a helpful, concise AI assistant."),
        HumanMessage(content=message.content)
    ]
    msg = cl.Message(content="")
    await msg.send()
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                await msg.stream_token(chunk.content) # type: ignore
        await msg.update()
    except Exception as e:
        await msg.stream_token(f"\n\n🚨 Error: {str(e)}")
        await msg.update()


# =====================================================================
# 2. PHASE 2: CUSTOM WEBSOCKET FOR FLUTTER
# =====================================================================
@fastapi_app.websocket("/api/chat/ws")
async def flutter_websocket_endpoint(websocket: WebSocket):
    # Accept the connection from the Flutter app
    await websocket.accept()
    print("📱 Flutter client connected to WebSocket!")
    
    try:
        while True:
            # 1. Wait for the user's message from Flutter
            user_text = await websocket.receive_text()
            print(f"Received from Flutter: {user_text}")
            
            # 2. Emit the "Thinking..." status (Matches PRD Phase 3 Integration)
            await websocket.send_json({"type": "status", "content": "consulting_tool"})
            
            # 3. Prepare the messages for Groq
            messages = [
                SystemMessage(content="You are a helpful AI assistant talking to a mobile user. Be concise."),
                HumanMessage(content=user_text)
            ]
            
            # 4. Tell Flutter the stream is starting
            await websocket.send_json({"type": "stream_start"})
            
            # 5. Stream the tokens directly to the Flutter socket
            try:
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        # Send each token as a tiny JSON packet
                        await websocket.send_json({"type": "token", "content": chunk.content})
                
                # 6. Tell Flutter the stream is finished
                await websocket.send_json({"type": "stream_end"})
                
            except Exception as e:
                print(f"Groq error: {e}")
                await websocket.send_json({"type": "error", "content": f"LLM Error: {str(e)}"})
                
    except WebSocketDisconnect:
        print("📱 Flutter client disconnected.")