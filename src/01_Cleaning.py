print("🚀 Cleaning pipeline started...")

"""
01_cleaning.py
NorthBridge Health – Data Cleaning Pipeline

This script:
- Loads the raw Excel dataset
- Cleans each sheet using modular helper functions
- Saves cleaned CSVs into Data/processed/
"""

import pandas as pd
import os

# -------------------------------------------------------------------
# 1. File paths
# -------------------------------------------------------------------

RAW_FILE = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\NorthBridgeDataset.xlsx"
OUTPUT_DIR = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Cleaning pipeline started...")
print(f"Loading raw file: {RAW_FILE}")


# -------------------------------------------------------------------
# 2. Generic cleaning helpers
# -------------------------------------------------------------------

def clean_dates(df, cols):
    """Convert listed columns to datetime safely."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_text(df, cols):
    """Lowercase + strip whitespace for categorical text fields."""
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
    return df


def fill_missing(df, fill_map):
    """Fill missing values for specified columns."""
    for col, value in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


# -------------------------------------------------------------------
# 3. Sheet-specific cleaning functions
# -------------------------------------------------------------------

def clean_tickets(df):
    """Clean the Tickets sheet."""
    df = clean_dates(df, ["CreatedAt", "FirstResponseAt", "ResolvedAt", "SLADueAt"])

    df = clean_text(df, ["Status", "Channel", "Description", "ResolutionNotes"])

    df = fill_missing(
        df,
        {
            "PriorityID": "unknown",
            "CategoryID": "unknown",
            "Channel": "unknown",
        },
    )

    return df


def clean_clients(df):
    """Clean the Clients sheet."""
    df = clean_text(df, ["ClientName", "ClientType", "Region", "ContractTier", "SLACreditClause"])
    df = clean_dates(df, ["ContractStartDate", "ContractEndDate"])
    return df


def clean_agents(df):
    """Clean the Agents sheet."""
    df = clean_text(df, ["FullName", "Role", "Specialisms", "Hub"])

    if "DailyCapacity" in df.columns:
        df["DailyCapacity"] = pd.to_numeric(df["DailyCapacity"], errors="coerce")

    return df


def clean_ticket_audit(df):
    """Clean the TicketAuditLog sheet."""
    df = clean_dates(df, ["ActionAt"])
    df = clean_text(df, ["ActionType", "PreviousValue", "NewValue", "Notes"])
    return df


def clean_escalation_log(df):
    """Clean the EscalationLog sheet."""
    df = clean_dates(df, ["EscalatedAt"])
    df = clean_text(df, ["EscalationReason"])
    return df


# -------------------------------------------------------------------
# 4. Load Excel file and clean each sheet
# -------------------------------------------------------------------

xlsx = pd.ExcelFile(RAW_FILE)
print("Sheets found:", xlsx.sheet_names)

# Load sheets
tickets_raw = pd.read_excel(xlsx, "Tickets")
clients_raw = pd.read_excel(xlsx, "Clients")
agents_raw = pd.read_excel(xlsx, "Agents")
audit_raw = pd.read_excel(xlsx, "TicketAuditLog")
escalations_raw = pd.read_excel(xlsx, "EscalationLog")

# Clean sheets
tickets = clean_tickets(tickets_raw)
clients = clean_clients(clients_raw)
agents = clean_agents(agents_raw)
audit = clean_ticket_audit(audit_raw)
escalations = clean_escalation_log(escalations_raw)

# -------------------------------------------------------------------
# 5. Save cleaned datasets
# -------------------------------------------------------------------

tickets.to_csv(os.path.join(OUTPUT_DIR, "tickets_clean.csv"), index=False)
clients.to_csv(os.path.join(OUTPUT_DIR, "clients_clean.csv"), index=False)
agents.to_csv(os.path.join(OUTPUT_DIR, "agents_clean.csv"), index=False)
audit.to_csv(os.path.join(OUTPUT_DIR, "audit_clean.csv"), index=False)
escalations.to_csv(os.path.join(OUTPUT_DIR, "escalations_clean.csv"), index=False)

print("\nCleaning completed successfully.")
print(f"Cleaned files saved to: {OUTPUT_DIR}")
