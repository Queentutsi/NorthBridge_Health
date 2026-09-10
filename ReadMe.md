NorthBridge Health Services Ltd — SLA Optimisation & Predictive Analytics Project

This repository contains the full end‑to‑end data engineering, analytics, forecasting, and machine learning work completed for NorthBridge Health Services Ltd, a UK healthcare operations company headquartered in Manchester. The project modernises their customer‑service ticketing workflows, centralises fragmented operational data, and introduces predictive SLA‑breach modelling to support proactive service management.

📌 Project Overview
NorthBridge supports over 680 healthcare clients and processes 11,000+ service tickets monthly, but rising demand exposed major inefficiencies in SLA management, backlog control, and workload balancing.
The organisation faced:

First‑response times rising to 38 hours (vs 8‑hour SLA for Priority 1)

Over 2,800 unresolved tickets during audits

Fragmented data across CRM, spreadsheets, and email

Manual ticket assignment and inconsistent SLA tracking

£214,000+ annual SLA‑related financial exposure

This project delivers a unified cloud data pipeline, real‑time operational dashboards, forecasting, and a machine‑learning SLA‑breach prediction model.

🎯 Project Objectives
1. Centralise Operational Ticket Data
Build a Snowflake‑based reporting environment using S3 → Airbyte → Snowflake ingestion.
Includes raw landing, cleaning, and promotion into a reporting schema.

2. SLA Monitoring & Escalation Visibility
Power BI dashboards highlight SLA timelines, at‑risk tickets, breach patterns, and escalation behaviour.

3. Real‑Time Operational Dashboards
Interactive reporting for backlog trends, workload distribution, capacity utilisation, and SLA performance.

4. Workload Balancing & Capacity Reporting
Daily agent‑level workload heatmaps, ticket complexity scoring, and team capacity utilisation metrics.

5. KPI Framework & Historical Tracking
Standardised DAX measures for SLA compliance, response times, resolution times, backlog trends, and escalation rates.

6. Demand & SLA Forecasting
Power BI time‑series forecasting for ticket volume, backlog growth, and SLA breach trends.

7. Predictive SLA Breach Modelling
A supervised ML model predicting ticket‑level SLA breach probability using only existing schema fields:

Tickets: PriorityID, CategoryID, Channel, CreatedAt

Clients: ContractTier, SLACreditClause

Agents: Hub, TeamID, DailyCapacity

Model stack: Logistic Regression baseline → XGBoost/LightGBM tuned model.

📐 Data Architecture
Source → Landing → Warehouse → Reporting
Amazon S3 — raw CSV/Excel landing

Airbyte — ingestion into Snowflake RAW

Snowflake SQL — cleaning, standardisation, deduplication

Snowflake REPORTING schema — analysis‑ready tables

Power BI — dashboards, forecasting, SLA risk visualisation

Python (Snowpark / Notebook) — ML model training & scoring

🧱 Core Data Model
Fact Table
Tickets — central operational fact table

“Tickets.SLABreached… training label for the SLA breach prediction model.”

Dimensions
Clients

Agents

TicketCategories

PriorityLevels

SLADefinitions

Relationships:
Tickets → Clients
Tickets → Agents
Tickets → TicketCategories
Tickets → PriorityLevels

🤖 Machine Learning Component
Model Goal
Predict the probability that a ticket will breach its SLA before it approaches its deadline.

Features Used
All features come from existing tables — no new ingestion required.

Model Workflow
Feature engineering

Logistic Regression baseline

Gradient boosting (XGBoost/LightGBM)

Validation using precision, recall, ROC‑AUC

Scoring open tickets

Power BI visualisation alongside rules‑based SLA flags

📊 Dashboards Delivered
Executive Summary Dashboard

Operational Drilldown Dashboard

SLA Risk Monitoring Dashboard (rules‑based + ML risk score)

Workload & Capacity Dashboard

Demand & SLA Forecasting Dashboard

📁 Repository Structure (Suggested)
Code
/data_pipeline/
    /s3_raw/
    /airbyte_configs/
    /snowflake_sql/

/reporting/
    /data_dictionary/
    /dax_measures/

/ml_model/
    feature_engineering.ipynb
    model_training.ipynb
    model_validation.ipynb
    scoring_pipeline.py

/powerbi/
    dashboards.pbix
    forecasting_notes.md

/docs/
    architecture_diagram.png
    sla_definition_matrix.pdf
    retraining_schedule.md

README.md
📚 Documentation Included
Snowflake reporting‑layer data dictionary

DAX measure library

SLA breach model documentation

Retraining schedule

Forecasting methodology

Pipeline & ingestion documentation