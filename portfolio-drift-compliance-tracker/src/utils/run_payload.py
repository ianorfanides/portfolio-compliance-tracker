import sys
import os
import json
import asyncio
from dotenv import load_dotenv

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.workflow import root_agent

load_dotenv()

async def main():
    # Load custom payload
    payload_str = """{ "data": { "aggregate_portfolio_value": 100000.0, "account_type": "Margin", "account_number": "MAR-99008877", "account_owner": "Alastair Sterling", "usd_cad_rate": 1.42, "individual_assets": [ { "ticker": "VCN", "exchange": "TSX", "currency_of_denomination": "CAD", "amount_in_cad": 22000.0, "description": "Vanguard FTSE Canada All Cap Index ETF (Canadian Equity Index Fund)" }, { "ticker": "IVV", "exchange": "NYSE", "currency_of_denomination": "USD", "amount_in_cad": 50000.0, "description": "iShares Core S&P 500 ETF (US Large Cap Equity Index)" }, { "ticker": "VAB", "exchange": "TSX", "currency_of_denomination": "CAD", "amount_in_cad": 1000.0, "description": "Vanguard Canadian Aggregate Bond Index ETF (Canadian Fixed Income)" } ] } }"""
    payload = json.loads(payload_str)
    
    session_service = InMemorySessionService()
    APP_NAME = "portfolio_compliance_app"
    USER_ID = "compliance_officer"
    SESSION_ID = "custom_scenario_session"
    
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
        parts=[types.Part(text=json.dumps(payload))]
    )
    
    print("Starting custom compliance scenario run...")
    paused_event = None
    
    async for event_out in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message_content
    ):
        if event_out.node_info:
            print(f"[Node: {event_out.node_info.path}] running...")
            
        is_req = False
        if event_out.content and event_out.content.parts:
            for part in event_out.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    is_req = True
                    
        if is_req or (event_out.content and "Do you approve" in str(event_out.content)):
            paused_event = event_out
            print("\n>>> Workflow paused: Waiting for human approval.")
            
    if paused_event:
        interrupt_id = None
        if paused_event.content and paused_event.content.parts:
            for part in paused_event.content.parts:
                if part.function_call and part.function_call.name == 'adk_request_input':
                    interrupt_id = part.function_call.id
                    
        if interrupt_id:
            decision = "approve"
            print(f"\n[Auto-HITL] Sending response: '{decision}' for interrupt: {interrupt_id}...")
            
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
                    
    # Print final outcome
    session_data = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print("\n--- FINAL SESSION EVENTS ---")
    for i, event in enumerate(session_data.events):
        print(f"Event {i}: Author={event.author}, Node={event.node_info.path if event.node_info else ''}")
        if event.output:
            print(f"  Output: {event.output}")

if __name__ == "__main__":
    asyncio.run(main())
