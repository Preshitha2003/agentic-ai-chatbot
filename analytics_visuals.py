

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

os.makedirs("reports/visuals", exist_ok=True)

def generate_chart(query: str):
    """
    Auto-detects chart type and visualizes relevant data from PostgreSQL.
    """

    q = query.lower()

    # 🔹 Step 1: Pick a relevant table (superstore preferred)
    table = "superstore"

    # 🔹 Step 2: Try to infer what the user wants to compare or plot
    if "region" in q:
        group_field = "Region"
    elif "category" in q:
        group_field = "Category"
    elif "sub-category" in q or "subcategory" in q:
        group_field = "Sub-Category"
    elif "state" in q:
        group_field = "State"
    elif "segment" in q:
        group_field = "Segment"
    elif "month" in q or "date" in q or "trend" in q or "time" in q:
        group_field = "Order Date"
    else:
        group_field = "Category"

    # 🔹 Step 3: Choose a metric
    if "profit" in q:
        metric = "Profit"
    elif "sales" in q or "revenue" in q or "income" in q:
        metric = "Sales"
    else:
        metric = "Sales"

    # 🔹 Step 4: Decide chart type automatically
    if "trend" in q or "time" in q or "growth" in q or group_field in ["Order Date"]:
        chart_type = "line"
    elif "share" in q or "percentage" in q or "distribution" in q:
        chart_type = "pie"
    else:
        chart_type = "bar"

    # 🔹 Step 5: SQL query to fetch data
    sql = text(f'SELECT "{group_field}", SUM("{metric}") AS value FROM {table} GROUP BY "{group_field}" ORDER BY value DESC LIMIT 10')
    df = pd.read_sql(sql, engine)

    if df.empty:
        raise ValueError("No data retrieved for chart.")

    # 🔹 Step 6: Clean / sort data
    df = df.dropna().sort_values(by="value", ascending=False)

    # 🔹 Step 7: Generate chart
    plt.figure(figsize=(8, 5))
    if chart_type == "bar":
        plt.bar(df[group_field], df["value"])
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "line":
        plt.plot(df[group_field], df["value"], marker="o")
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "pie":
        plt.pie(df["value"], labels=df[group_field], autopct="%1.1f%%")

    plt.title(f"{metric} by {group_field}")
    plt.xlabel(group_field)
    plt.ylabel(metric)
    plt.tight_layout()

    # 🔹 Step 8: Save chart
    filename = f"reports/visuals/{chart_type}_chart_{group_field}_vs_{metric}.png"
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"🎨 Detected chart type: {chart_type.upper()}")
    print(f"✅ Chart saved at: {filename}")

    return filename



