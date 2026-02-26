import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Free Fast LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",  # type: ignore
    api_key=os.environ.get("GROQ_API_KEY")
) # type: ignore

@cl.step(name="processing_request", type="tool")
async def process_intent():
    """Simulates the agent using a tool or analyzing the request."""
    cl.context.current_step.input = "Analyzing user text..." # type: ignore
    cl.context.current_step.output = "Intent understood. Generating response." # type: ignore
    return "Success"

@cl.on_message
async def main(message: cl.Message):
    # 1. Run the dummy step
    await process_intent()
    
    # 2. Prepare the messages
    messages = [
        SystemMessage(content="You are a helpful, concise AI assistant. Reply directly and naturally to the user's message."),
        HumanMessage(content=message.content)
    ]
    
    # 3. Create an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()
    
    # 4. Try to stream, but catch any errors and display them!
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                await msg.stream_token(chunk.content) # type: ignore
        await msg.update()
        
    except Exception as e:
        # If Groq throws an error, print it directly in the chat window
        error_msg = f"\n\n🚨 **Error communicating with Groq:** {str(e)}"
        await msg.stream_token(error_msg)
        await msg.update()