import asyncio
import os
import sys
import base64
import json
from dotenv import load_dotenv

# Ensure the root directory is in sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.workflow import root_agent

# Load environment variables from .env file
load_dotenv()


# --- Generic Test Portfolios ---

# Test Case 1: Portfolio in compliance (Plain JSON)
# Total value is $100,000 CAD.
# All allocations are within 3% of their targets.
in_compliance_payload = {
    "aggregate_portfolio_value": 100000.0,
    "account_type": "TFSA",
    "account_number": "TFSA-887766",
    "account_owner": "John Doe",
    "individual_assets": [
        {"ticker": "VCN", "exchange": "TSX", "amount": 22000.0, "description": "Vanguard FTSE Canada All Cap Index ETF"},
        {"ticker": "VTI", "exchange": "NYSE", "amount": 18978.10, "description": "Vanguard Total Stock Market ETF"}, # 18978.10 * 1.37 = ~$26,000 CAD
        {"ticker": "VIU", "exchange": "TSX", "amount": 15000.0, "description": "Vanguard FTSE Developed All Cap ex North America Index ETF"},
        {"ticker": "VEE", "exchange": "TSX", "amount": 7000.0, "description": "Vanguard FTSE Emerging Markets Common Shares"},
        {"ticker": "VAB", "exchange": "TSX", "amount": 25000.0, "description": "Vanguard Canadian Aggregate Bond Index ETF"},
        {"ticker": "CASH", "exchange": "CAD", "amount": 5000.0, "description": "Cash holdings in Canadian Dollars"}
    ]
}

# Test Case 2: Portfolio out of compliance (Base64 Encoded - simulates Pub/Sub trigger)
# US stocks are heavily over-allocated, and Canadian/EM equities are under-allocated.
out_of_compliance_payload = {
    "aggregate_portfolio_value": 100000.0,
    "account_type": "Margin",
    "account_number": "MAR-112233",
    "account_owner": "Jane Smith",
    "individual_assets": [
        {"ticker": "VCN", "exchange": "TSX", "amount": 10000.0, "description": "Vanguard FTSE Canada All Cap Index ETF"}, # 10% (Target: 22%) -> BUY
        {"ticker": "VTI", "exchange": "NYSE", "amount": 35000.0, "description": "Vanguard Total Stock Market ETF"}, # 35000 * 1.37 = $47,950 CAD -> 47.95% (Target: 26%) -> SELL
        {"ticker": "VIU", "exchange": "TSX", "amount": 10000.0, "description": "Vanguard FTSE Developed All Cap ex North America Index ETF"}, # 10% (Target: 15%) -> BUY
        {"ticker": "VEE", "exchange": "TSX", "amount": 2000.0, "description": "Vanguard FTSE Emerging Markets Common Shares"}, # 2% (Target: 7%) -> BUY
        {"ticker": "VAB", "exchange": "TSX", "amount": 25000.0, "description": "Vanguard Canadian Aggregate Bond Index ETF"}, # 25% (Target: 25%) -> In Compliance
        {"ticker": "CASH", "exchange": "CAD", "amount": 5050.0, "description": "Cash holdings in Canadian Dollars"} # 5.05% (Target: 5%) -> In Compliance
    ]
}

# Encode Jane's payload to base64
jane_json_str = json.dumps(out_of_compliance_payload)
jane_base64_str = base64.b64encode(jane_json_str.encode("utf-8")).decode("utf-8")

event_in_compliance = {"data": in_compliance_payload}
event_out_of_compliance = {"data": jane_base64_str}


async def run_compliance_check(event: dict, test_name: str, human_input_value: str = None):
    print(f"\n==================================================")
    print(f"RUNNING TEST: {test_name}")
    print(f"==================================================")
    
    session_service = InMemorySessionService()
    APP_NAME = "portfolio_compliance_app"
    USER_ID = "compliance_officer"
    SESSION_ID = f"session_{test_name.lower().replace(' ', '_')}"
    
    # Create the session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Wrap the event payload as the user message
    message_content = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps(event))]
    )
    
    # Step 1: Initial Run
    print("Initiating workflow...")
    paused_event = None
    req_msg = None
    
    async for event_out in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message_content
    ):
        if event_out.node_info:
            print(f"[Node: {event_out.node_info.path}] running...")
            
        # Check if event contains the RequestInput function call structure
        is_req = False
        if event_out.content and event_out.content.parts:
            for part in event_out.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    is_req = True
                    if part.function_call.args:
                        req_msg = part.function_call.args.get("message")
        
        # If the runner is waiting for input, it yields a RequestInput event
        if is_req or (event_out.content and "Do you approve" in str(event_out.content)):
            paused_event = event_out
            print("\n>>> Workflow paused: Waiting for human approval.")
            if req_msg:
                print(f"Prompt to Human:\n{req_msg}")
    
    # Step 2: Resume if paused and human input is provided
    if paused_event and human_input_value:
        # Extract the interrupt ID from the paused event's FunctionCall
        interrupt_id = None
        if paused_event.content and paused_event.content.parts:
            for part in paused_event.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    interrupt_id = part.function_call.id
                    
        if interrupt_id:
            print(f"\nResuming workflow with human decision: '{human_input_value}' for interrupt: {interrupt_id}...")
            
            # Create a FunctionResponse part to resolve the adk_request_input interrupt
            resume_part = types.Part(
                function_response=types.FunctionResponse(
                    id=interrupt_id,
                    name='adk_request_input',
                    response={'result': human_input_value}
                )
            )
            
            resume_message = types.Content(
                role="user",
                parts=[resume_part]
            )
            
            async for event_out in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=resume_message
            ):
                if event_out.node_info:
                    print(f"[Node: {event_out.node_info.path}] running...")
        else:
            print("\nError: Could not find active interrupt ID in paused event.")

    # Inspect the final session events
    session_data = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"\n--- SESSION EVENTS FOR {test_name} ---")
    for idx, e in enumerate(session_data.events):
        node_path = e.node_info.path if e.node_info else "None"
        print(f"Event {idx}: Author={e.author}, Node={node_path}")
        if e.output:
            print(f"  Output: {e.output}")
        if e.content:
            # Safely extract text content, function call, or function response details
            text_val = ""
            if e.content.parts:
                part = e.content.parts[0]
                if part.text:
                    text_val = part.text
                elif part.function_call:
                    text_val = f"FunctionCall: {part.function_call.name}({part.function_call.args})"
                elif part.function_response:
                    text_val = f"FunctionResponse: {part.function_response.name}(id={part.function_response.id}, response={part.function_response.response})"
            
            if text_val:
                print(f"  Content: {text_val[:150]}...")
            else:
                print(f"  Content: {str(e.content)[:150]}...")
    print("--------------------------------------\n")


async def main():
    # Ensure GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") != "1":
        print("WARNING: GOOGLE_API_KEY is not set.")
        return

    # Test 1: In Compliance (John Doe)
    await run_compliance_check(event_in_compliance, "John Doe - In Compliance")

    # Test 2: Out of Compliance (Jane Smith)
    await run_compliance_check(event_out_of_compliance, "Jane Smith - Out of Compliance", "approve")

    # Load and run new playground JSON tests
    playground_dir = os.path.join(os.path.dirname(__file__), "../playground_tests")
    
    # Test 3: Compliant Portfolio JSON (Clarissa Vance)
    compliant_path = os.path.join(playground_dir, "compliant_portfolio.json")
    if os.path.exists(compliant_path):
        with open(compliant_path, "r") as f:
            compliant_event = json.load(f)
        await run_compliance_check(compliant_event, "Playground Compliant Portfolio")

    # Test 4: Drifted Portfolio JSON (Alastair Sterling)
    drifted_path = os.path.join(playground_dir, "drifted_portfolio.json")
    if os.path.exists(drifted_path):
        with open(drifted_path, "r") as f:
            drifted_event = json.load(f)
        await run_compliance_check(drifted_event, "Playground Drifted Portfolio", "approve")

    # Test 5: Adversarial Injection Portfolio JSON (Malicious Tester)
    injection_path = os.path.join(playground_dir, "adversarial_injection_portfolio.json")
    if os.path.exists(injection_path):
        with open(injection_path, "r") as f:
            injection_event = json.load(f)
        await run_compliance_check(injection_event, "Playground Adversarial Injection Portfolio")


if __name__ == "__main__":
    asyncio.run(main())
