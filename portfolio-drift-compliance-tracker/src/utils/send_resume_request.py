import urllib.request
import json

def main():
    req_body = {
        "subscription": "projects/my-gcp-project/subscriptions/compliance-check-sub",
        "interrupt_id": "d9f3f39b-cc2c-4977-b5c3-5ab3129815c5",
        "decision": "approve"
    }

    print("Sending resume POST request to http://localhost:8080/resume ...")
    req = urllib.request.Request(
        "http://localhost:8080/resume",
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
