

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# === Create SQLAlchemy engine ===
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# === Helper Functions ===
def load_csv_to_postgres(file_path, table_name):
    """Reads any CSV and loads it into PostgreSQL."""
    df = pd.read_csv(file_path, encoding="latin1")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"✅ Loaded {table_name} ({len(df)} rows)")


def verify_table_counts():
    """Prints row counts of all tables in the database."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        )
        tables = [row[0] for row in result]
        for t in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"📊 {t}: {count} rows")


# === Main Execution ===
def main():
    data_dir = "data"

    # Get all CSV files dynamically
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"🗂 Found CSV files: {csv_files}")

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        table_name = os.path.splitext(csv_file)[0].lower()  # e.g. Superstore.csv → superstore
        load_csv_to_postgres(file_path, table_name)

    # Optional: create derived tables (only if 'superstore' exists)
    with engine.connect() as conn:
        existing_tables = [
            row[0]
            for row in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            )
        ]

    if "superstore" in existing_tables:
        superstore = pd.read_sql("SELECT * FROM superstore", engine)

        # Derive logical tables
        products = superstore[["Product ID", "Product Name", "Category", "Sub-Category"]].drop_duplicates()
        regions = superstore[["Region", "State", "City"]].drop_duplicates()
        calendar = superstore[["Order Date", "Ship Date"]].drop_duplicates()

        # Store them
        products.to_sql("products", engine, if_exists="replace", index=False)
        regions.to_sql("regions", engine, if_exists="replace", index=False)
        calendar.to_sql("calendar", engine, if_exists="replace", index=False)

        print("🧩 Derived tables created: products, regions, calendar")

    # Verify
    verify_table_counts()


if __name__ == "__main__":
    main()
