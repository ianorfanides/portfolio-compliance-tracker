import base64
import json
from typing import Dict, Any, List
from config import TARGET_ALLOCATIONS, DRIFT_THRESHOLD, USD_TO_CAD

def mask_account_number(acc_num: str) -> str:
    """Masks all but the last 4 characters of the account number for PII protection."""
    acc_str = str(acc_num).strip()
    if len(acc_str) <= 4:
        return "*" * len(acc_str)
    return "*" * (len(acc_str) - 4) + acc_str[-4:]


def parse_input_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parses the incoming portfolio ingestion event.
    Handles both base64-encoded Pub/Sub data and plain JSON data under the 'data' key.
    """
    data_payload = event.get("data")
    if not data_payload:
        raise ValueError("Missing 'data' key in the incoming event.")

    # Decode base64 if it's a string and looks like base64
    if isinstance(data_payload, str):
        try:
            # Try decoding as base64
            decoded_bytes = base64.b64decode(data_payload, validate=True)
            data_str = decoded_bytes.decode("utf-8")
            data = json.loads(data_str)
        except Exception:
            # If base64 decoding fails, try to parse it directly as a JSON string
            try:
                data = json.loads(data_payload)
            except Exception as e:
                raise ValueError(f"Failed to parse 'data' string as JSON: {e}")
    elif isinstance(data_payload, dict):
        data = data_payload
    else:
        raise ValueError("Invalid format for 'data' key. Expected JSON string, base64 string, or dict.")

    # Extract required fields and apply PII masking
    parsed_portfolio = {
        "aggregate_portfolio_value": float(data.get("aggregate_portfolio_value", 0.0)),
        "account_type": data.get("account_type", "Unknown"),
        "account_number": mask_account_number(data.get("account_number", "Unknown")),
        "account_owner": data.get("account_owner", "Unknown"),
        "usd_cad_rate": float(data.get("usd_cad_rate", USD_TO_CAD)),
        "assets": []
    }

    # Extract individual assets
    assets_raw = data.get("individual_assets") or data.get("assets", [])
    for asset in assets_raw:
        amount_in_cad = asset.get("amount_in_cad")
        amount = float(asset.get("amount", 0.0))
        
        parsed_portfolio["assets"].append({
            "ticker": asset.get("ticker", "").strip(),
            "exchange": asset.get("exchange", "").strip(),
            "amount": amount,  # Value in original currency (if amount_in_cad is not specified)
            "description": asset.get("description", "").strip(),
            "currency_of_denomination": asset.get("currency_of_denomination", "CAD").strip(),
            "amount_in_cad": float(amount_in_cad) if amount_in_cad is not None else None
        })

    return parsed_portfolio


def calculate_compliance(portfolio: Dict[str, Any], classifications: Dict[str, str]) -> Dict[str, Any]:
    """Performs the compliance calculations, applying currency conversion and rebalancing rules."""
    assets = portfolio["assets"]
    usd_cad_rate = portfolio.get("usd_cad_rate", USD_TO_CAD)
    
    # 1. Calculate asset values in CAD (base currency)
    assets_in_cad = []
    total_calculated_value = 0.0
    
    for asset in assets:
        ticker = asset["ticker"]
        exchange = asset["exchange"].upper()
        amount = asset["amount"]
        description = asset["description"]
        amount_in_cad = asset.get("amount_in_cad")
        
        # If amount_in_cad is provided in the JSON, use it directly as the CAD value.
        # Otherwise, calculate it using the USD_CAD rate if it trades on a US exchange.
        if amount_in_cad is not None:
            val_in_cad = amount_in_cad
        else:
            is_us_exchange = exchange in ["NYSE", "NASDAQ", "AMEX", "US"]
            rate = usd_cad_rate if is_us_exchange else 1.0
            val_in_cad = amount * rate
        
        # Get category from classifications (default to Cash if empty/unknown)
        category = classifications.get(ticker, "Cash" if "CASH" in ticker.upper() or not ticker else "Cash")
        
        assets_in_cad.append({
            "ticker": ticker,
            "exchange": exchange,
            "amount_original": amount,
            "value_cad": val_in_cad,
            "category": category,
            "description": description
        })
        total_calculated_value += val_in_cad

    # Always prioritize the precise mathematically calculated total from assets to prevent LLM sum hallucinations
    if total_calculated_value > 0:
        total_portfolio_value = total_calculated_value
    else:
        total_portfolio_value = portfolio.get("aggregate_portfolio_value") or 1.0

    # 2. Group by category
    category_totals = {cat: 0.0 for cat in TARGET_ALLOCATIONS.keys()}
    category_tickers = {cat: [] for cat in TARGET_ALLOCATIONS.keys()}
    
    for asset in assets_in_cad:
        cat = asset["category"]
        if cat not in category_totals:
            cat = "Cash"  # Fallback
        category_totals[cat] += asset["value_cad"]
        category_tickers[cat].append(asset["ticker"])

    # 3. Compute drifts and variances
    drifts = []
    needs_approval = False
    rebalance_recommendations = []

    for cat, target_pct in TARGET_ALLOCATIONS.items():
        current_val = category_totals[cat]
        current_pct = current_val / total_portfolio_value
        variance = current_pct - target_pct
        abs_variance = abs(variance)
        
        tickers_in_scope = category_tickers[cat]
        
        if abs_variance < DRIFT_THRESHOLD:
            status = "in portfolio"
            action = "No action needed"
        else:
            status = "OUT OF COMPLIANCE"
            needs_approval = True
            if variance < 0:
                action = f"BUY {cat} (Under-allocated by {abs_variance*100:.2f}%)"
                rebalance_recommendations.append({
                    "category": cat,
                    "action": "BUY",
                    "variance": abs_variance,
                    "tickers_in_scope": tickers_in_scope,
                    "message": f"Recommend BUYING {cat} assets to cover a {abs_variance*100:.2f}% under-allocation. Tickers in scope: {', '.join(tickers_in_scope)}"
                })
            else:
                action = f"SELL/TRADE {cat} (Over-allocated by {abs_variance*100:.2f}%)"
                rebalance_recommendations.append({
                    "category": cat,
                    "action": "SELL",
                    "variance": abs_variance,
                    "tickers_in_scope": tickers_in_scope,
                    "message": f"Recommend SELLING/TRADING {cat} assets to cover a {abs_variance*100:.2f}% over-allocation. Tickers in scope: {', '.join(tickers_in_scope)}"
                })

        drifts.append({
            "category": cat,
            "current_value_cad": round(current_val, 2),
            "current_percentage": round(current_pct, 4),
            "target_percentage": round(target_pct, 4),
            "variance": round(variance, 4),
            "status": status,
            "action": action,
            "tickers": tickers_in_scope
        })

    return {
        "account_owner": portfolio["account_owner"],
        "account_number": portfolio["account_number"],
        "account_type": portfolio["account_type"],
        "total_portfolio_value_cad": round(total_portfolio_value, 2),
        "drifts": drifts,
        "needs_approval": needs_approval,
        "rebalance_recommendations": rebalance_recommendations
    }
