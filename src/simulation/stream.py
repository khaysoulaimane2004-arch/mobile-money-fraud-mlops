"""
Transaction Simulator
Sends fake mobile money transactions to the API
and gradually injects drift over time.
"""

import requests
import random
import time
import json
import os
import csv
from datetime import datetime


API_URL   = "http://localhost:8080/predict"
LOG_FILE  = "reports/drift/transactions_log.csv"


def generate_normal_transaction():
    """Generate a legitimate-looking transaction."""
    transaction_type = random.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"])
    amount           = random.lognormvariate(9, 1.5)
    old_balance      = random.lognormvariate(10, 2)
    new_balance      = max(0, old_balance - amount)

    return {
        "type"          : transaction_type,
        "amount"        : round(amount, 2),
        "oldbalanceOrg" : round(old_balance, 2),
        "newbalanceOrig": round(new_balance, 2),
        "oldbalanceDest": round(random.lognormvariate(9, 2), 2),
        "newbalanceDest": round(random.lognormvariate(9, 2) + amount, 2),
    }


def generate_drifted_transaction(drift_factor=0.5):
    """
    Generate a transaction with drift injected.
    As drift_factor increases (0 to 1), transactions
    look more and more like fraud patterns.
    """
    # Drifted transactions have higher amounts
    amount      = random.lognormvariate(11 + drift_factor * 2, 1.5)
    old_balance = random.lognormvariate(11 + drift_factor, 2)

    # More likely to be TRANSFER or CASH_OUT as drift increases
    if random.random() < drift_factor:
        transaction_type = random.choice(["TRANSFER", "CASH_OUT"])
        new_balance      = 0  # empties the account
    else:
        transaction_type = random.choice(["PAYMENT", "CASH_IN", "DEBIT"])
        new_balance      = max(0, old_balance - amount)

    return {
        "type"          : transaction_type,
        "amount"        : round(amount, 2),
        "oldbalanceOrg" : round(old_balance, 2),
        "newbalanceOrig": round(new_balance, 2),
        "oldbalanceDest": round(random.lognormvariate(9, 2), 2),
        "newbalanceDest": round(random.lognormvariate(9, 2) + amount, 2),
    }


def log_transaction(transaction, response, drift_factor):
    """Save each transaction and its prediction to a CSV file."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp", "type", "amount",
                "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest",
                "is_fraud", "fraud_probability",
                "drift_factor"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            transaction["type"],
            transaction["amount"],
            transaction["oldbalanceOrg"],
            transaction["newbalanceOrig"],
            transaction["oldbalanceDest"],
            transaction["newbalanceDest"],
            response.get("is_fraud", False),
            response.get("fraud_probability", 0),
            round(drift_factor, 2)
        ])


def run_simulation(
    n_transactions = 500,
    drift_start    = 200,
    delay_seconds  = 0.1
):
    """
    Run the simulation.
    
    - First `drift_start` transactions are normal
    - After that, drift is gradually injected
    - drift_factor goes from 0 to 1 linearly
    """
    print(f"Starting simulation — {n_transactions} transactions")
    print(f"Drift starts at transaction {drift_start}")
    print(f"Logging to {LOG_FILE}")
    print("-" * 50)

    fraud_count = 0
    error_count = 0

    for i in range(n_transactions):

        # Calculate drift factor
        if i < drift_start:
            drift_factor = 0.0
            transaction  = generate_normal_transaction()
        else:
            drift_factor = (i - drift_start) / (n_transactions - drift_start)
            transaction  = generate_drifted_transaction(drift_factor)

        # Send to API
        try:
            r = requests.post(API_URL, json=transaction, timeout=5)
            response = r.json()

            if response.get("is_fraud"):
                fraud_count += 1

            log_transaction(transaction, response, drift_factor)

            # Print progress every 50 transactions
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{n_transactions}] "
                      f"Drift: {drift_factor:.2f} | "
                      f"Frauds detected: {fraud_count} | "
                      f"Errors: {error_count}")

        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"Error at transaction {i}: {e}")

        time.sleep(delay_seconds)

    print("-" * 50)
    print(f"Simulation complete ✓")
    print(f"Total transactions : {n_transactions}")
    print(f"Frauds detected    : {fraud_count}")
    print(f"Errors             : {error_count}")
    print(f"Log saved to       : {LOG_FILE}")


if __name__ == "__main__":
    run_simulation(
        n_transactions = 500,
        drift_start    = 200,
        delay_seconds  = 0.05
    )