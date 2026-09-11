"""
03_Feature_engineering.py
NorthBridge Health – Feature Engineering Pipeline
"""

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# 1. File paths
# -------------------------------------------------------------------

BASE_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\processed"
OUTPUT_FILE = os.path.join(BASE_PATH, "model_dataset.csv")

print("🚀 Feature engineering pipeline started...")
print(f"Loading cleaned datasets from: {BASE_PATH}")

# -------------------------------------------------------------------
# 2. Load cleaned datasets
# -------------------------------------------------------------------

tickets = pd.read_csv(os.path.join(BASE_PATH, "tickets_clean.csv"))
clients = pd.read_csv(os.path.join(BASE_PATH, "clients_clean.csv"))
agents = pd.read_csv(os.path.join(BASE_PATH, "agents_clean.csv"))
audit = pd.read_csv(os.path.join(BASE_PATH, "audit_clean.csv"))
escalations = pd.read_csv(os.path.join(BASE_PATH, "escalations_clean.csv"))

print("📄 Loaded datasets:")
print("tickets:", tickets.shape)
print("clients:", clients.shape)
print("agents:", agents.shape)
print("audit:", audit.shape)
print("escalations:", escalations.shape)

# -------------------------------------------------------------------
# 3. Convert timestamps
# -------------------------------------------------------------------

datetime_cols = ["CreatedAt", "FirstResponseAt", "ResolvedAt", "SLADueAt"]
for col in datetime_cols:
    tickets[col] = pd.to_datetime(tickets[col], errors="coerce")

# -------------------------------------------------------------------
# 4. Escalations merge (safe)
# -------------------------------------------------------------------

tickets["TicketID"] = tickets["TicketID"].astype(str).str.strip()
escalations["TicketID"] = escalations["TicketID"].astype(str).str.strip()

esc_counts = (
    escalations.groupby("TicketID")
    .size()
    .reset_index(name="NumEscalations")
)

tickets = tickets.merge(esc_counts, on="TicketID", how="left")
tickets["NumEscalations"] = tickets["NumEscalations"].fillna(0)
tickets["IsEscalatedFlag"] = (tickets["NumEscalations"] > 0).astype(int)

# -------------------------------------------------------------------
# 5. Time-based engineered features
# -------------------------------------------------------------------

tickets["ResponseTimeHours"] = (
    (tickets["FirstResponseAt"] - tickets["CreatedAt"]).dt.total_seconds() / 3600
)

tickets["ResolutionTimeHours"] = (
    (tickets["ResolvedAt"] - tickets["CreatedAt"]).dt.total_seconds() / 3600
)

tickets["SLAMarginHours"] = (
    (tickets["ResolvedAt"] - tickets["SLADueAt"]).dt.total_seconds() / 3600
)

tickets["ResponseDelayedFlag"] = (tickets["ResponseTimeHours"] > 24).astype(int)
tickets["ResolutionDelayedFlag"] = (tickets["ResolutionTimeHours"] > 72).astype(int)

tickets["ResponseTimeBucket"] = pd.cut(
    tickets["ResponseTimeHours"],
    bins=[-1, 4, 24, 72, 9999],
    labels=["very_fast", "fast", "slow", "very_slow"]
)

tickets["ResolutionTimeBucket"] = pd.cut(
    tickets["ResolutionTimeHours"],
    bins=[-1, 24, 72, 168, 9999],
    labels=["quick", "moderate", "slow", "very_slow"]
)

# -------------------------------------------------------------------
# 6. Merge clients + agents
# -------------------------------------------------------------------

tickets = tickets.merge(clients, on="ClientID", how="left", suffixes=("", "_client"))
tickets = tickets.merge(
    agents,
    left_on="AssignedAgentID",
    right_on="AgentID",
    how="left",
    suffixes=("", "_agent")
)

# -------------------------------------------------------------------
# 7. Behavioural aggregates
# -------------------------------------------------------------------

agent_agg = (
    tickets.groupby("AssignedAgentID")
    .agg(
        AgentTicketCount=("TicketID", "count"),
        AgentSlaRate=("SLABreached", "mean"),
        AgentAvgResponse=("ResponseTimeHours", "mean"),
        AgentAvgResolution=("ResolutionTimeHours", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(agent_agg, on="AssignedAgentID", how="left")

client_agg = (
    tickets.groupby("ClientID")
    .agg(
        ClientTicketCount=("TicketID", "count"),
        ClientSlaRate=("SLABreached", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(client_agg, on="ClientID", how="left")

category_agg = (
    tickets.groupby("CategoryName")
    .agg(
        CategoryTicketCount=("TicketID", "count"),
        CategorySlaRate=("SLABreached", "mean"),
        CategoryAvgResolution=("ResolutionTimeHours", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(category_agg, on="CategoryName", how="left")

priority_agg = (
    tickets.groupby("PriorityID")
    .agg(
        PriorityTicketCount=("TicketID", "count"),
        PrioritySlaRate=("SLABreached", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(priority_agg, on="PriorityID", how="left")

channel_agg = (
    tickets.groupby("Channel")
    .agg(
        ChannelTicketCount=("TicketID", "count"),
        ChannelSlaRate=("SLABreached", "mean"),
        ChannelAvgResponse=("ResponseTimeHours", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(channel_agg, on="Channel", how="left")

status_agg = (
    tickets.groupby("Status")
    .agg(
        StatusTicketCount=("TicketID", "count"),
        StatusSlaRate=("SLABreached", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(status_agg, on="Status", how="left")

tickets["IsPendingClientFlag"] = tickets["Status"].str.contains("pending", case=False, na=False).astype(int)

# -------------------------------------------------------------------
# 8. Time trend features
# -------------------------------------------------------------------

tickets["Month"] = tickets["CreatedAt"].dt.to_period("M").astype(str)
tickets["Season"] = tickets["CreatedAt"].dt.month % 12 // 3 + 1

month_agg = (
    tickets.groupby("Month")
    .agg(
        MonthTicketCount=("TicketID", "count"),
        MonthSlaRate=("SLABreached", "mean"),
    )
    .reset_index()
)
tickets = tickets.merge(month_agg, on="Month", how="left")

# -------------------------------------------------------------------
# 9. Text-based features
# -------------------------------------------------------------------

tickets["Description"] = tickets["Description"].fillna("").str.lower()

keywords = {
    "PatientFlag": ["patient"],
    "AppointmentFlag": ["appointment", "schedule", "reschedule"],
    "InsurerFlag": ["insurer", "insurance"],
    "PaymentFlag": ["payment", "charge", "billing"],
    "InvoiceFlag": ["invoice"],
    "ReferralFlag": ["referral"],
    "HoursFlag": ["hours"],
    "ConfirmationFlag": ["confirmation"],
    "UpdatedFlag": ["updated"],
}

def contains_any(text, words):
    return int(any(w in text for w in words))

for flag, words in keywords.items():
    tickets[flag] = tickets["Description"].apply(lambda x: contains_any(x, words))

tickets["DescriptionLength"] = tickets["Description"].str.len()
tickets["DescriptionWordCount"] = tickets["Description"].str.split().str.len()

# -------------------------------------------------------------------
# 10. Save modelling dataset
# -------------------------------------------------------------------

tickets.to_csv(OUTPUT_FILE, index=False)

print("✅ Feature engineering completed successfully.")
print(f"Model dataset saved to: {OUTPUT_FILE}")
