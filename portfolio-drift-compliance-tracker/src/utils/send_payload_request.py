import urllib.request
import json
import base64

def main():
    payload_data = {
        "aggregate_portfolio_value": 100000.0,
        "account_type": "Margin",
        "account_number": "MAR-99008877",
        "account_owner": "Alastair Sterling",
        "usd_cad_rate": 1.42,
        "individual_assets": [
            {"ticker": "VCN", "exchange": "TSX", "currency_of_denomination": "CAD", "amount_in_cad": 22000.0, "description": "Vanguard FTSE Canada All Cap Index ETF"},
            {"ticker": "IVV", "exchange": "NYSE", "currency_of_denomination": "USD", "amount_in_cad": 50000.0, "description": "iShares Core S&P 500 ETF"},
            {"ticker": "VAB", "exchange": "TSX", "currency_of_denomination": "CAD", "amount_in_cad": 1000.0, "description": "Vanguard Canadian Aggregate Bond Index ETF"}
        ]
    }

    data_str = json.dumps(payload_data)
    base64_data = base64.b64encode(data_str.encode("utf-8")).decode("utf-8")

    req_body = {
        "subscription": "projects/my-gcp-project/subscriptions/compliance-check-sub",
        "message": {
            "data": base64_data
        }
    }

    print("Sending POST request to http://localhost:8080/ ...")
    req = urllib.request.Request(
        "http://localhost:8080/",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_content = response.read().decode("utf-8")
            res_json = json.loads(res_content)
            print("\nResponse Received:")
            print(json.dumps(res_json, indent=2))
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
