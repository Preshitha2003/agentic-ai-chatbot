
# table_agent.py
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from fuzzywuzzy import process

load_dotenv()

# -------------------------
# Database Connection
# -------------------------
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# -------------------------
# Table Agent Logic
# -------------------------
def table_agent(query: str):
    """
    Enhanced Table Agent:
    - Handles general comparisons (categories, regions, segments)
    - Handles specific product comparisons (multi-product queries)
    """

    q = query.lower()
    table = "superstore"

    try:
        # --- Load minimal data for context ---
        with engine.connect() as conn:
            df_products = pd.read_sql(text(f'SELECT DISTINCT "Product Name" FROM {table};'), conn)

        product_list = df_products["Product Name"].dropna().tolist()

        # --- Detect product names in the query (using fuzzy matching) ---
        mentioned_products = []
        for word in q.replace(",", " ").split():
            match, score = process.extractOne(word, product_list)
            if score > 85 and match not in mentioned_products:
                mentioned_products.append(match)

        # --- Case 1: Multiple products mentioned (side-by-side comparison) ---
        if len(mentioned_products) >= 2:
            placeholders = ', '.join([f"'{p}'" for p in mentioned_products])
            sql = f"""
            SELECT "Product Name", "Category", ROUND(SUM("Sales")::numeric, 2) AS Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Profit, 
                   ROUND(AVG("Discount")::numeric, 2) AS Avg_Discount, 
                   SUM("Quantity") AS Total_Quantity
            FROM {table}
            WHERE "Product Name" IN ({placeholders})
            GROUP BY "Product Name", "Category"
            ORDER BY Sales DESC;
            """

            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)

            if df.empty:
                return "⚠️ No matching products found for comparison."

            # Display as Markdown table
            markdown_table = df.to_markdown(index=False)
            return f"📊 Product Comparison Table:\n\n{markdown_table}"

        # --- Case 2: General comparison requests ---
        elif "category" in q:
            sql = f"""
            SELECT "Category", 
                   ROUND(SUM("Sales")::numeric, 2) AS Total_Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Total_Profit
            FROM {table}
            GROUP BY "Category"
            ORDER BY Total_Sales DESC;
            """
        elif "region" in q:
            sql = f"""
            SELECT "Region", 
                   ROUND(SUM("Sales")::numeric, 2) AS Total_Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Total_Profit
            FROM {table}
            GROUP BY "Region"
            ORDER BY Total_Sales DESC;
            """
        elif "segment" in q:
            sql = f"""
            SELECT "Segment", 
                   ROUND(SUM("Sales")::numeric, 2) AS Total_Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Total_Profit
            FROM {table}
            GROUP BY "Segment"
            ORDER BY Total_Sales DESC;
            """
        elif "product" in q or "top" in q:
            sql = f"""
            SELECT "Product Name", 
                   ROUND(SUM("Sales")::numeric, 2) AS Total_Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Total_Profit
            FROM {table}
            GROUP BY "Product Name"
            ORDER BY Total_Sales DESC
            LIMIT 10;
            """
        else:
            sql = f"""
            SELECT "Category", 
                   ROUND(SUM("Sales")::numeric, 2) AS Total_Sales, 
                   ROUND(SUM("Profit")::numeric, 2) AS Total_Profit
            FROM {table}
            GROUP BY "Category"
            ORDER BY Total_Sales DESC;
            """

        # Execute the SQL for general comparison
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        markdown_table = df.to_markdown(index=False)
        return f"📊 Comparison Table:\n\n{markdown_table}"

    except Exception as e:
        return f"❌ Table Agent error: {str(e)}"
