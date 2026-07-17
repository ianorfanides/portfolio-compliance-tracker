import os
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compliance_app")

# --- Telemetry Configuration ---
# Set otel_to_cloud to False explicitly
otel_to_cloud = False
logger.info(f"Telemetry configured: otel_to_cloud={otel_to_cloud}")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.workflow import root_agent

# Initialize FastAPI web service
app = FastAPI(title="Ambient Portfolio Compliance Web Service")

@app.get("/")
async def welcome():
    """Welcome/Health Check endpoint for GET requests."""
    return {
        "status": "RUNNING",
        "service": "Ambient Portfolio Compliance Web Service",
        "endpoints": {
            "POST /": "Accepts Pub/Sub push trigger messages",
            "POST /resume": "Resumes a paused compliance approval workflow"
        }
    }

# Shared state session service to persist session event histories across API calls
session_service = InMemorySessionService()

def normalize_subscription(subscription_path: str) -> str:
    """Normalizes a fully-qualified subscription path to a short name.
    Example: 'projects/my-project/subscriptions/my-subscription' -> 'my-subscription'
    """
    if not subscription_path:
        return "default-subscription"
    return subscription_path.split("/")[-1]


@app.post("/")
async def pubsub_trigger(request: Request):
    """POST endpoint to accept Pub/Sub trigger messages and run the compliance workflow."""
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request body as JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    logger.info(f"Received Pub/Sub trigger request: {json.dumps(body)}")
    
    # 1. Normalize the subscription path
    subscription_path = body.get("subscription", "projects/local/subscriptions/default-subscription")
    normalized_sub = normalize_subscription(subscription_path)
    logger.info(f"Normalized subscription: {subscription_path} -> {normalized_sub}")
    
    # 2. Extract message object and its inner data
    message = body.get("message")
    if not message or not isinstance(message, dict):
        logger.error("Missing 'message' object in Pub/Sub payload")
        raise HTTPException(status_code=400, detail="Missing 'message' object in payload")
        
    data_payload = message.get("data")
    if not data_payload:
        logger.error("Missing 'data' field in 'message' object")
        raise HTTPException(status_code=400, detail="Missing 'data' field in 'message' object")
        
    # Re-wrap into the event payload format expected by the workflow parser
    event_payload = {"data": data_payload}
    
    APP_NAME = "portfolio_compliance_app"
    USER_ID = "pubsub_trigger"
    SESSION_ID = f"session_{normalized_sub}"
    
    # Ensure the session exists in the shared service
    session_exists = None
    try:
        session_exists = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    except Exception:
        pass

    if session_exists is None:
        logger.info(f"Creating new session: {SESSION_ID}")
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    else:
        logger.info(f"Re-using existing session: {SESSION_ID}")
    
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    message_content = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps(event_payload))]
    )
    
    # 3. Run the ADK workflow
    logger.info(f"Executing workflow for session {SESSION_ID}...")
    final_output = None
    paused_event = None
    
    async for event_out in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message_content
    ):
        if event_out.node_info:
            logger.info(f"[Node: {event_out.node_info.path}] running...")
            
        is_req = False
        if event_out.content and event_out.content.parts:
            for part in event_out.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    is_req = True
                    
        if is_req or (event_out.content and "Do you approve" in str(event_out.content)):
            paused_event = event_out
            logger.info(">>> Workflow paused: Waiting for human approval.")
            
        if event_out.output:
            final_output = event_out.output

    if paused_event:
        # Extract prompt message and active interrupt ID
        interrupt_id = None
        req_msg = None
        if paused_event.content and paused_event.content.parts:
            for part in paused_event.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    interrupt_id = part.function_call.id
                    if part.function_call.args:
                        req_msg = part.function_call.args.get("message")
                        
        logger.info(f"Workflow paused. Session: {SESSION_ID}, Interrupt ID: {interrupt_id}")
        return {
            "status": "PAUSED_WAITING_FOR_APPROVAL",
            "session_id": SESSION_ID,
            "interrupt_id": interrupt_id,
            "prompt": req_msg,
            "message": "The compliance check is out of compliance and has paused for human-in-the-loop approval."
        }
        
    logger.info(f"Workflow finished. Session: {SESSION_ID}, Output: {final_output}")
    return {
        "status": "COMPLETED",
        "session_id": SESSION_ID,
        "output": final_output
    }


@app.post("/resume")
async def resume_workflow(request: Request):
    """POST endpoint to resume a paused compliance workflow with the human decision."""
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request body as JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    subscription_path = body.get("subscription", "projects/local/subscriptions/default-subscription")
    normalized_sub = normalize_subscription(subscription_path)
    
    interrupt_id = body.get("interrupt_id")
    decision = body.get("decision", "approve").strip()
    
    if not interrupt_id:
        logger.error("Missing 'interrupt_id' in resume payload")
        raise HTTPException(status_code=400, detail="Missing 'interrupt_id' in resume payload")
        
    APP_NAME = "portfolio_compliance_app"
    USER_ID = "pubsub_trigger"
    SESSION_ID = f"session_{normalized_sub}"
    
    logger.info(f"Resuming session {SESSION_ID} for interrupt {interrupt_id} with decision '{decision}'...")
    
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Construct the FunctionResponse part matching the adk_request_input interrupt
    resume_part = types.Part(
        function_response=types.FunctionResponse(
            id=interrupt_id,
            name='adk_request_input',
            response={'result': decision}
        )
    )
    
    resume_message = types.Content(
        role="user",
        parts=[resume_part]
    )
    
    final_output = None
    async for event_out in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=resume_message
    ):
        if event_out.node_info:
            logger.info(f"[Node: {event_out.node_info.path}] running...")
        if event_out.output:
            final_output = event_out.output
            
    logger.info(f"Workflow finished after resume. Session: {SESSION_ID}, Output: {final_output}")
    return {
        "status": "COMPLETED",
        "session_id": SESSION_ID,
        "output": final_output
    }


@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """GET endpoint to inspect the detailed execution events of a compliance session."""
    APP_NAME = "portfolio_compliance_app"
    USER_ID = "pubsub_trigger"
    try:
        session_data = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Failed to find session {session_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    if session_data is None:
        logger.warning(f"Session {session_id} not found in memory service storage")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    events = []
    for idx, event in enumerate(session_data.events):
        text_val = ""
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if part.text:
                text_val = part.text
            elif part.function_call:
                text_val = f"FunctionCall: {part.function_call.name}({part.function_call.args})"
            elif part.function_response:
                text_val = f"FunctionResponse: {part.function_response.name}(id={part.function_response.id}, response={part.function_response.response})"
                
        events.append({
            "event_index": idx,
            "author": event.author,
            "node_path": event.node_info.path if event.node_info else None,
            "output": event.output,
            "content": text_val if text_val else str(event.content) if event.content else None
        })
        
    return {
        "session_id": session_id,
        "event_count": len(events),
        "events": events
    }
