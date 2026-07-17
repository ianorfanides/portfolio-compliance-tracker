# config.py

# Model for asset classification
MODEL_NAME = "gemini-3.1-flash-lite"

# Variance threshold for rebalancing (3% of total portfolio)
DRIFT_THRESHOLD = 0.03

# Target allocations for each asset category
TARGET_ALLOCATIONS = {
    "Cdn": 0.22,   # Canadian Equity
    "US": 0.26,    # US Equity
    "EAFE": 0.15,  # EAFE (Europe, Australasia, Far East)
    "EM": 0.07,    # Emerging Markets
    "FI": 0.25,    # Fixed Income
    "Cash": 0.05   # Cash
}

# Currency Exchange Rate (USD to CAD)
USD_TO_CAD = 1.37
