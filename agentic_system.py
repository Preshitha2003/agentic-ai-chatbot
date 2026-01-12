"""
Agentic framework (single-file) for:
- Router (intent classification)
- SQL Agent (safe SQL execution against your PostgreSQL)
- RAG Agent (Pinecone + local summarizer)
- ML Agent (small forecasting demo)
- Report Agent (generate PDF)
- Short-term memory (in-memory)

Drop this into your project root and run with: python agentic_system.py
"""

import os
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
from sqlalchemy import create_engine, text, inspect
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression
import numpy as np

from analytics_visuals import generate_chart
from table_agent import table_agent



load_dotenv()


from sentence_transformers import SentenceTransformer, util

# Load lightweight intent model
intent_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example reference intents
intent_labels = {
    "SQL": ["show total sales", "count how many", "average revenue", "sum of profit", "group by region"],
    "RAG": ["explain policy", "describe process", "what are responsibilities", "summarize report", "kpis in guide"],
    "ML": ["forecast", "predict next quarter", "estimate growth", "trend analysis"],
    "REPORT": ["create pdf", "generate report", "save analysis", "build report"]
}
intent_embeddings = {k: intent_model.encode(v, convert_to_tensor=True) for k, v in intent_labels.items()}


# -------------------------
# Config / Connections
# -------------------------
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embedding and summarizer (reuse)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -------------------------
# Short-term memory (very simple)
# -------------------------
class ShortTermMemory:
    def __init__(self, max_items=5):
        self.max = max_items
        self.items = []  # list of (timestamp, query, agent, answer)

    def add(self, query, agent, answer):
        self.items.append((time.time(), query, agent, answer))
        if len(self.items) > self.max:
            self.items.pop(0)

    def show(self):
        for ts, q, a, ans in self.items:
            print(f"- [{time.ctime(ts)}] {q} -> {a}")

memory = ShortTermMemory(max_items=6)

# -------------------------
# Utility: inspect DB schema and whitelist names
# -------------------------
inspector = inspect(engine)
public_tables = [t for t in inspector.get_table_names(schema='public')]
table_columns = {}
for t in public_tables:
    cols = [c['name'] for c in inspector.get_columns(t)]
    table_columns[t] = cols

# Helper to validate table/column names (prevents SQL injection)
def validate_table_and_columns(table, columns):
    table = table.lower()
    if table not in table_columns:
        return False, f"Table '{table}' not found in database."
    for c in columns:
        if c not in table_columns[table]:
            return False, f"Column '{c}' not found in table '{table}'."
    return True, None

# -------------------------
# SQL Agent
# -------------------------
def sql_agent(nl_query):
    """
    Very lightweight natural-language -> SQL mapping using keyword detection.
    Supports basic aggregates: SUM, AVG, COUNT, GROUP BY.
    """
    q = nl_query.lower()

    # simple detection of table - find a known table name in the query
    detected_table = None
    for t in public_tables:
        if t in q:
            detected_table = t
            break
    # fallback: if 'superstore' exists use it; else first table
    if not detected_table:
        detected_table = 'superstore' if 'superstore' in public_tables else public_tables[0]

    # detect aggregate
    agg = None
    if any(word in q for word in ['sum', 'total', 'total of', 'total sales', 'total revenue']):
        agg = 'SUM'
        agg_col = 'Sales' if 'sales' in table_columns.get(detected_table, []) or 'sales' in q else None
        if not agg_col:
            # try common choices
            candidates = [c for c in table_columns.get(detected_table, []) if c.lower() in ('sales','revenue','amount','total')]
            agg_col = candidates[0] if candidates else table_columns[detected_table][0]
    elif any(word in q for word in ['average','avg','mean']):
        agg = 'AVG'
        agg_col = 'Sales' if 'Sales' in table_columns.get(detected_table, []) else table_columns[detected_table][0]
    elif any(word in q for word in ['count','how many','number of']):
        agg = 'COUNT'
        agg_col = '*'
    else:
        # fallback to select top lines
        agg = None

    # detect group by field (region, category, state, product, customer)
    group_field = None
    for candidate in ['region','state','city','category','product id','product name','customerid','customer id','sub-category','sub_category','category']:
        if candidate in q:
            # map to actual DB column name
            # simple mapping:
            oracle = candidate.replace(' ','_').replace('-','_')
            # find matching column name (case-sensitive in schema)
            for col in table_columns.get(detected_table, []):
                if col.lower().replace(' ','_') == oracle:
                    group_field = col
                    break
            if group_field:
                break

    # Build SQL
    if agg:
        if agg_col == '*':
            select_clause = f"COUNT(*) as total"
        else:
            # ensure agg_col exists
            colname = None
            for c in table_columns[detected_table]:
                if c.lower() == agg_col.lower() or c.lower().replace('_','') == agg_col.lower().replace('_',''):
                    colname = c
                    break
            if not colname:
                colname = table_columns[detected_table][0]
            # select_clause = f"{agg}({colname}) as value"
            select_clause = f'{agg}("{colname}") as value'
        sql = f"SELECT {select_clause}"
        if group_field:
            
            sql += f', "{group_field}" FROM {detected_table} GROUP BY "{group_field}" ORDER BY value DESC LIMIT 10'
        else:
            sql += f" FROM {detected_table} LIMIT 100"
    else:
        # fallback: show sample rows
        sql = f"SELECT * FROM {detected_table} LIMIT 10"

    # Validate column names used (group_field)
    if group_field:
        ok, err = validate_table_and_columns(detected_table, [group_field])
        if not ok:
            return f"SQL agent: validation error - {err}"

    # Execute
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
        text_out = f"SQL Agent executed:\n{sql}\n\nResults:\n{df.to_string(index=False)}"
        memory.add(nl_query, 'SQL', text_out[:1000])
        return text_out
    except Exception as e:
        return f"SQL Agent error: {e}"

# -------------------------
# RAG Agent
# -------------------------
def rag_agent(nl_query, top_k=3):
    query_vec = embed_model.encode(nl_query).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    if not results.get('matches'):
        return "RAG Agent: No matching document context found."
    # Combine and summarize
    contexts = [m['metadata']['text'] for m in results['matches']]
    combined = " ".join(contexts)
    # Limit combined length to summarizer capacity
    combined_short = combined[:3000]
    summary = summarizer(combined_short, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
    ans = f"RAG Agent retrieved {len(contexts)} chunks. Summary:\n\n{summary}"
    memory.add(nl_query, 'RAG', ans)
    return ans

# -------------------------
# ML Agent (mini forecasting example)
# -------------------------
def ml_agent(nl_query):
    # This is a lightweight demo: forecast quarterly revenue using linear regression
    # We'll use 'superstore' if available and aggregate monthly total 'Sales'
    table = 'superstore' if 'superstore' in public_tables else None
    if not table:
        return "ML Agent: No 'superstore' table available for forecasting."

    try:
        df = pd.read_sql(f"SELECT \"Order Date\", \"Sales\" FROM {table} WHERE \"Order Date\" IS NOT NULL", engine)
        # convert order date to month ordinal
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df = df.dropna(subset=['Order Date'])
        df['month_idx'] = (df['Order Date'].dt.year * 12 + df['Order Date'].dt.month)
        agg = df.groupby('month_idx', as_index=False)['Sales'].sum()
        # require at least 6 points
        if len(agg) < 6:
            return "ML Agent: Not enough historical data to forecast."

        X = agg['month_idx'].values.reshape(-1,1)
        y = agg['Sales'].values
        model = LinearRegression().fit(X, y)
        next_month = np.array([[agg['month_idx'].max() + 1]])
        pred = model.predict(next_month)[0]
        ans = f"ML Agent forecast: Next-month revenue estimate = {pred:,.2f} (linear model based on monthly totals)"
        memory.add(nl_query, 'ML', ans)
        return ans
    except Exception as e:
        return f"ML Agent error: {e}"

# -------------------------
# Report Agent - generate a small PDF
# -------------------------
def report_agent(nl_query, data_text):
    """
    Creates a short PDF report from `data_text` (string).
    Filename is returned.
    """
    outname = f"reports/ai_report_{int(time.time())}.pdf"
    os.makedirs("reports", exist_ok=True)
    c = canvas.Canvas(outname, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "AI Generated Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 60, f"Query: {nl_query}")
    text_obj = c.beginText(40, height - 90)
    text_obj.setFont("Helvetica", 10)
    # split lines
    for line in data_text.splitlines():
        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_obj.textLine(chunk)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    memory.add(nl_query, 'REPORT', f"Saved {outname}")
    # return f"Report created: {outname}"
    # return f"reports/ai_report_{int(time.time())}.pdf"
    return outname


# -------------------------
# Visualization Agent
# -------------------------
def visualization_agent(query):
    """
    Handles data visualization queries (bar, line, pie).
    Detects the right chart type and generates it from PostgreSQL.
    """
    try:
        chart_path = generate_chart(query)
        return f"📊 Visualization generated! Check the saved chart at:\n{chart_path}"
    except Exception as e:
        return f"❌ Visualization error: {str(e)}"

# added on 10 Nov 2025 for visualization routing


# -------------------------
# Smarter Hybrid Router (Visualization + Report + Semantic fallback)
# -------------------------
def route_query(nl_query):
    q = nl_query.lower().strip()

    # 🟢 Step 1: Visualization-first detection
    if any(k in q for k in [
        "chart", "graph", "visualize", "visualization", "plot", "draw", "display",
        "show me a chart", "show me a graph", "visual representation", "create chart", "generate chart"
    ]):
        return "VISUAL", visualization_agent(nl_query)

    # 🧠 Step 2: Semantic similarity detection
    query_emb = intent_model.encode(q, convert_to_tensor=True)
    scores = {label: float(util.cos_sim(query_emb, emb).max()) for label, emb in intent_embeddings.items()}
    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]
    print(f"\n🧭 Routing decision: {best_label} (confidence={confidence:.3f})")

    # 🧾 Step 3: Detect explicit report generation intent
    if any(k in q for k in [
        "create report", "generate report", "build report", "make report",
        "pdf report", "generate pdf", "save report", "summary report"
    ]):
        print("📝 Generating report...")

        if any(k in q for k in ['top', 'by', 'highest', 'lowest', 'top 5', 'top 10']):
            sql_out = sql_agent(q)
            pdf_path = report_agent(nl_query, sql_out)
        else:
            rag_out = rag_agent(q)
            pdf_path = report_agent(nl_query, rag_out)

        return "REPORT", f"📄 Report created successfully!\nSaved at: {pdf_path}"
   
    # 🧩 Step 3.5: Context override — if *referring to* a report → RAG Agent
    if any(k in q for k in [
        "according to the report", "in the report", "from the report",
        "as per the report", "based on the report", "mentioned in the report",
        "as per the summary", "from the document", "in the document"
    ]):
        return "RAG", rag_agent(nl_query)
    
    
    # 🟣 Step 4: Table comparison detection        #added on 11 Nov 2025 for table agent
    if any(k in q for k in [
        "compare", "comparison", "difference between", "contrast", 
        "versus", "vs", "table of", "comparison table",
        "list top", "top products", "top categories", "show table", "show comparison",
        "best", "highest", "top region", "top category", "performed best", "most profitable", "lowest"
    ]):
        return "TABLE", table_agent(nl_query)


    # 🧩 Step 5: Fallback if semantic confidence is low
    if confidence < 0.35:
        if any(k in q for k in ['forecast', 'predict', 'trend', 'estimate', 'projection']):
            best_label = 'ML'
        elif any(k in q for k in [
            'sum', 'total', 'average', 'avg', 'count', 'how many', 'top', 'bottom',
            'rank', 'max', 'min', 'group by', 'by region', 'by state', 'by category',
            'top 5', 'top 10', 'sales amount', 'revenue', 'profit', 'income'
        ]):
            best_label = 'SQL'
        else:
            best_label = 'RAG'

    # 🧮 Step 6: Execute the corresponding agent
    if best_label == 'SQL':
        return 'SQL', sql_agent(nl_query)
    elif best_label == 'RAG':
        return 'RAG', rag_agent(nl_query)
    elif best_label == 'ML':
        return 'ML', ml_agent(nl_query)
    elif best_label == 'REPORT':
        # Safety catch (should rarely be needed)
        print("📝 Generating fallback report...")
        rag_out = rag_agent(q)
        pdf_path = report_agent(nl_query, rag_out)
        return 'REPORT', f"📄 Report created successfully!\nSaved at: {pdf_path}"
    else:
        return 'RAG', rag_agent(nl_query)




# -------------------------
# Interactive loop
# -------------------------
def main():
    print("🧩 Agentic System ready. Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ('exit','quit'):
            print("Goodbye!")
            break
        agent_name, response = route_query(user)
        print(f"\n[{agent_name} Agent response]\n{response}\n")
        print("-"*80 + "\n")

if __name__ == "__main__":
    main()



