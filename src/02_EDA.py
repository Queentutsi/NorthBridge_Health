"""
02_EDA.py
Exploratory Data Analysis for NorthBridge Health Ticketing System

This script:
- Loads cleaned datasets
- Generates all EDA charts (distribution, trends, correlations)
- Provides modular functions for reuse
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------------------
# 1. Load cleaned datasets
# -------------------------------------------------------------------

BASE_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\processed"

tickets = pd.read_csv(os.path.join(BASE_PATH, "tickets_clean.csv"))
clients = pd.read_csv(os.path.join(BASE_PATH, "clients_clean.csv"))
agents = pd.read_csv(os.path.join(BASE_PATH, "agents_clean.csv"))
audit = pd.read_csv(os.path.join(BASE_PATH, "audit_clean.csv"))
escalations = pd.read_csv(os.path.join(BASE_PATH, "escalations_clean.csv"))

print("Loaded datasets:")
print("tickets:", tickets.shape)
print("clients:", clients.shape)
print("agents:", agents.shape)
print("audit:", audit.shape)
print("escalations:", escalations.shape)

# -------------------------------------------------------------------
# 2. NHS colour palette (your preference)
# -------------------------------------------------------------------

nhs_colors = {
    "nhs_dark_blue": "#005EB8",
    "nhs_light_blue": "#41B6E6",
    "warm_yellow": "#FFB81C",
    "orange": "#F46A25",
    "light_green": "#78BE20",
    "grey": "#425563",
}

sns.set(style="whitegrid")

# -------------------------------------------------------------------
# 3. Helper: Save charts (optional)
# -------------------------------------------------------------------

OUTPUT_CHARTS = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\charts"
os.makedirs(OUTPUT_CHARTS, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(OUTPUT_CHARTS, f"{name}.png"), dpi=300, bbox_inches="tight")

# -------------------------------------------------------------------
# 4. Category Distribution
# -------------------------------------------------------------------

def plot_category_distribution(df):
    category_counts = df["CategoryName"].value_counts()
    plt.figure(figsize=(12,6))
    category_counts.plot(kind="barh", color=nhs_colors["nhs_dark_blue"])
    plt.title("CategoryName Distribution", fontsize=18)
    plt.xlabel("Number of Tickets")
    plt.ylabel("CategoryName")
    plt.tight_layout()
    save_fig("category_distribution")
    plt.show()

# -------------------------------------------------------------------
# 5. Channel Distribution
# -------------------------------------------------------------------

def plot_channel_distribution(df):
    channel_counts = df["Channel"].value_counts()
    plt.figure(figsize=(12,6))
    channel_counts.plot(kind="barh", color=nhs_colors["nhs_light_blue"])
    plt.title("Channel Distribution", fontsize=18)
    plt.xlabel("Number of Tickets")
    plt.ylabel("Channel")
    plt.tight_layout()
    save_fig("channel_distribution")
    plt.show()

# -------------------------------------------------------------------
# 6. Status Distribution
# -------------------------------------------------------------------

def plot_status_distribution(df):
    status_counts = df["Status"].value_counts()
    plt.figure(figsize=(12,6))
    status_counts.plot(kind="barh", color=nhs_colors["warm_yellow"])
    plt.title("Status Distribution", fontsize=18)
    plt.xlabel("Number of Tickets")
    plt.ylabel("Status")
    plt.tight_layout()
    save_fig("status_distribution")
    plt.show()

# -------------------------------------------------------------------
# 7. Workload Distribution by Agent
# -------------------------------------------------------------------

def plot_agent_workload(df):
    agent_counts = df["AssignedAgentID"].value_counts()
    plt.figure(figsize=(14,6))
    agent_counts.plot(kind="bar", color=nhs_colors["light_green"])
    plt.title("Workload Distribution by Agent", fontsize=18)
    plt.xlabel("Agent ID")
    plt.ylabel("Tickets")
    plt.tight_layout()
    save_fig("agent_workload")
    plt.show()

# -------------------------------------------------------------------
# 8. Time-based features (response/resolution)
# -------------------------------------------------------------------

tickets["CreatedAt"] = pd.to_datetime(tickets["CreatedAt"], errors="coerce")
tickets["FirstResponseAt"] = pd.to_datetime(tickets["FirstResponseAt"], errors="coerce")
tickets["ResolvedAt"] = pd.to_datetime(tickets["ResolvedAt"], errors="coerce")

tickets["ResponseTimeHours"] = (
    (tickets["FirstResponseAt"] - tickets["CreatedAt"]).dt.total_seconds() / 3600
)

tickets["ResolutionTimeHours"] = (
    (tickets["ResolvedAt"] - tickets["CreatedAt"]).dt.total_seconds() / 3600
)

# -------------------------------------------------------------------
# 9. Response Time Box Plot
# -------------------------------------------------------------------

def plot_response_time(df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x=df["ResponseTimeHours"], color=nhs_colors["nhs_light_blue"])
    plt.title("ResponseTimeHours Distribution", fontsize=18)
    plt.xlabel("ResponseTimeHours")
    plt.tight_layout()
    save_fig("response_time_boxplot")
    plt.show()

# -------------------------------------------------------------------
# 10. Resolution Time Box Plot
# -------------------------------------------------------------------

def plot_resolution_time(df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x=df["ResolutionTimeHours"], color=nhs_colors["orange"])
    plt.title("ResolutionTimeHours Distribution", fontsize=18)
    plt.xlabel("ResolutionTimeHours")
    plt.tight_layout()
    save_fig("resolution_time_boxplot")
    plt.show()

# -------------------------------------------------------------------
# 11. SLA Breach Rate by Agent
# -------------------------------------------------------------------

def plot_sla_by_agent(df):
    agent_sla = df.groupby("AssignedAgentID")["SLABreached"].mean().sort_values()
    plt.figure(figsize=(14,6))
    agent_sla.plot(kind="bar", color=nhs_colors["orange"])
    plt.title("SLA Breach Rate by Agent", fontsize=18)
    plt.xlabel("Agent ID")
    plt.ylabel("Breach Rate")
    plt.tight_layout()
    save_fig("sla_by_agent")
    plt.show()

# -------------------------------------------------------------------
# 12. SLA Breach Rate by Client
# -------------------------------------------------------------------

def plot_sla_by_client(df):
    client_sla = df.groupby("ClientID")["SLABreached"].mean().sort_values()
    plt.figure(figsize=(14,6))
    client_sla.plot(kind="bar", color=nhs_colors["warm_yellow"])
    plt.title("SLA Breach Rate by Client", fontsize=18)
    plt.xlabel("Client ID")
    plt.ylabel("Breach Rate")
    plt.tight_layout()
    save_fig("sla_by_client")
    plt.show()

# -------------------------------------------------------------------
# 13. Monthly Ticket Volume & SLA Breaches
# -------------------------------------------------------------------

def plot_monthly_trends(df):
    df["Month"] = df["CreatedAt"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month").agg(
        Tickets=("TicketID", "count"),
        Breaches=("SLABreached", "sum")
    )
    plt.figure(figsize=(14,6))
    plt.plot(monthly.index, monthly["Tickets"], label="Ticket Volume", color=nhs_colors["nhs_dark_blue"])
    plt.plot(monthly.index, monthly["Breaches"], label="SLA Breaches", color=nhs_colors["orange"])
    plt.title("Monthly Ticket Volume & SLA Breaches", fontsize=18)
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig("monthly_trends")
    plt.show()

# -------------------------------------------------------------------
# 14. Correlation Heatmap
# -------------------------------------------------------------------

def plot_correlation(df):
    numeric_cols = [
        "ClientID", "CategoryID", "PriorityID", "AssignedAgentID",
        "SLABreached", "ResponseTimeHours", "ResolutionTimeHours", "NumUpdates"
    ]
    corr_df = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_df, annot=True, cmap="Blues")
    plt.title("Correlation Heatmap", fontsize=18)
    plt.tight_layout()
    save_fig("correlation_heatmap")
    plt.show()

# -------------------------------------------------------------------
# 15. Run all EDA functions
# -------------------------------------------------------------------

plot_category_distribution(tickets)
plot_channel_distribution(tickets)
plot_status_distribution(tickets)
plot_agent_workload(tickets)
plot_response_time(tickets)
plot_resolution_time(tickets)
plot_sla_by_agent(tickets)
plot_sla_by_client(tickets)
plot_monthly_trends(tickets)
plot_correlation(tickets)

print("EDA completed successfully.")
