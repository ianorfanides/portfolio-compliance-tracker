import csv
import json
import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.workflow import root_agent

load_dotenv()

async def run_csv_workflow(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    print(f"Reading CSV file: {csv_path}")
    
    # Group assets by AccountNumber
    accounts = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc_num = row.get("AccountNumber", "").strip()
            if not acc_num:
                continue
            if acc_num not in accounts:
                accounts[acc_num] = {
                    "aggregate_portfolio_value": 0.0,
                    "account_type": row.get("AccountType", "").strip(),
                    "account_number": acc_num,
                    "account_owner": row.get("AccountOwner", "").strip(),
                    "individual_assets": []
                }
            
            amount = float(row.get("Amount", "0.0").strip() or 0.0)
            accounts[acc_num]["individual_assets"].append({
                "ticker": row.get("Ticker", "").strip(),
                "exchange": row.get("Exchange", "").strip(),
                "amount": amount,
                "description": row.get("Description", "").strip()
            })
            accounts[acc_num]["aggregate_portfolio_value"] += amount

    # Run workflow for each account
    for acc_num, portfolio in accounts.items():
        print(f"\n==================================================")
        print(f"Processing Account: {acc_num} Owner: {portfolio['account_owner']}")
        print(f"==================================================")
        
        # Compile into Pub/Sub event format (under "data" key)
        event_payload = {
            "data": portfolio
        }
        
        # Setup runner
        session_service = InMemorySessionService()
        APP_NAME = "portfolio_compliance_app"
        USER_ID = "compliance_officer"
        SESSION_ID = f"csv_session_{acc_num}"
        
        # Create the session in the session service
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
        
        message_content = types.Content(
            role="user",
            parts=[types.Part(text=json.dumps(event_payload))]
        )
        
        print("Initiating compliance workflow...")
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
            
            # Identify if workflow is paused on RequestInput
            if is_req or (event_out.content and "Do you approve" in str(event_out.content)):
                paused_event = event_out
                print("\n>>> Workflow paused: Waiting for human approval.")
                if req_msg:
                    print(f"Prompt to Human:\n{req_msg}")
        
        if paused_event:
            # Extract the interrupt ID from the paused event's FunctionCall
            interrupt_id = None
            if paused_event.content and paused_event.content.parts:
                for part in paused_event.content.parts:
                    if part.function_call and part.function_call.name == 'adk_request_input':
                        interrupt_id = part.function_call.id
            
            if not interrupt_id:
                print("Error: Could not find active interrupt ID in paused event.")
                continue
                
            # Auto-approve the rebalancing trades to show end-to-end execution
            decision = "approve"
            print(f"\n[Auto-HITL] Sending response: '{decision}' for interrupt: {interrupt_id}...")
            
            # Create a FunctionResponse part matching the adk_request_input call
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
            
            async for event_out in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=resume_message
            ):
                if event_out.node_info:
                    print(f"[Node: {event_out.node_info.path}] running...")
                    
        # Print final session status and events
        print("\n--- FINAL SESSION EVENTS ---")
        session_data = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        for i, event in enumerate(session_data.events):
            content_str = str(event.content)
            if len(content_str) > 150:
                content_str = content_str[:150] + "..."
            print(f"Event {i}: Author={event.author}, Node={event.node_info.path if event.node_info else ''}")
            if event.output:
                print(f"  Output: {event.output}")
            if event.content:
                # Safely extract text content or function call details
                text_val = ""
                if event.content.parts:
                    part = event.content.parts[0]
                    if part.text:
                        text_val = part.text
                    elif part.function_call:
                        text_val = f"FunctionCall: {part.function_call.name}({part.function_call.args})"
                    elif part.function_response:
                        text_val = f"FunctionResponse: {part.function_response.name}(id={part.function_response.id}, response={part.function_response.response})"
                
                if text_val:
                    print(f"  Content: {text_val[:200]}...")
                else:
                    print(f"  Content: {content_str}")
        print("--------------------------------------\n")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "test_portfolio.csv"
    asyncio.run(run_csv_workflow(csv_file))
