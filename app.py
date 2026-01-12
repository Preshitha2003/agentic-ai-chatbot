# # app.py
# import streamlit as st
# import traceback
# import pandas as pd
# import os

# # import your backend modules
# import agentic_system as agent
# import table_agent
# import analytics_visuals
# from db_connection import engine  # optional: used for raw SQL execution

# st.set_page_config(page_title="Agentic AI — Core Dashboard", layout="wide")
# st.title("🤖 AI Knowledge Assistant")
# st.markdown("Chat + Visuals + Table Comparison + Short-term Memory. Uses your existing `agentic_system.py` for routing.")

# # Session state
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Sidebar
# st.sidebar.header("Navigation")
# mode = st.sidebar.radio("Choose page", ["Chat", "Visualization", "SQL Runner", "Table Comparison", "Memory"])

# st.sidebar.markdown("---")
# st.sidebar.write("Env checks:")
# try:
#     num_tables = len(agent.public_tables) if hasattr(agent, "public_tables") else "N/A"
# except Exception:
#     num_tables = "N/A"
# st.sidebar.write(f"Postgres tables: {num_tables}")
# pc_status = "configured" if (hasattr(agent, "PINECONE_API_KEY") and agent.PINECONE_API_KEY) else "not configured"
# st.sidebar.write(f"Pinecone: {pc_status}")

# # Helper to render chat history
# def render_history():
#     if not st.session_state.history:
#         st.info("No chat history yet. Ask a question in Chat tab.")
#         return
#     for role, text in reversed(st.session_state.history):
#         if role == "You":
#             st.markdown(f"**🧑‍💻 You:** {text}")
#         else:
#             st.markdown(f"**🤖 Bot:** {text}")

# # --- Chat ---
# if mode == "Chat":
#     st.header("💬 Chat (RAG / SQL / ML / Report / Table / Visualization)")
#     q = st.text_input("Ask your question:", key="chat_input")
#     if st.button("Send"):
#         if not q.strip():
#             st.warning("Please type a question.")
#         else:
#             try:
#                 label, resp = agent.route_query(q)
#             except Exception as e:
#                 label = "ERROR"
#                 resp = f"Exception calling route_query: {e}\n\n{traceback.format_exc()}"
#             # store in history
#             st.session_state.history.append(("You", q))
#             # display depending on type
#             st.markdown(f"**[{label}]**")
#             # if response is very long, show as code block for readability
#             if isinstance(resp, str) and (len(resp) > 500 or "\n" in resp):
#                 st.code(resp)
#             else:
#                 st.write(resp)
#             st.session_state.history.append(("Bot", resp))

#     st.markdown("---")
#     st.subheader("Chat history (most recent first)")
#     render_history()

# # --- Visualization ---
# elif mode == "Visualization":
#     st.header("📈 Visualization (auto-detect)")
#     st.markdown("Enter an instruction like: 'Show sales by region as a bar chart' or 'plot profit trend by month'.")
#     vis_q = st.text_input("Visualization instruction:", key="vis_input")
#     if st.button("Generate visualization"):
#         if not vis_q.strip():
#             st.warning("Enter an instruction.")
#         else:
#             try:
#                 out = agent.visualization_agent(vis_q)
#                 st.success("Visualization agent response:")
#                 st.write(out)
#                 st.info("If chart file path returned, check `reports/visuals/` in project folder.")
#             except Exception as e:
#                 st.error(f"Visualization agent error: {e}\n{traceback.format_exc()}")

#     st.markdown("---")
#     st.write("Quick preview of latest visuals folder (if any):")
#     visuals_dir = "reports/visuals"
#     if os.path.isdir(visuals_dir):
#         files = sorted([f for f in os.listdir(visuals_dir) if f.endswith((".png", ".jpg"))], reverse=True)
#         if files:
#             latest = files[0]
#             st.image(os.path.join(visuals_dir, latest), caption=latest, use_column_width=True)
#             if len(files) > 1:
#                 st.write("Other saved visuals:")
#                 for f in files[1:6]:
#                     st.write(f"- {f}")
#         else:
#             st.info("No saved visuals yet. Generate one using the form above.")
#     else:
#         st.info("No visuals folder found. Generate a chart to create it.")

# # --- SQL Runner ---
# elif mode == "SQL Runner":
#     st.header("🧾 SQL Runner (direct execution)")
#     st.markdown("Use this to run exploratory SQL (read-only recommended).")
#     sql = st.text_area("Enter SQL (example: SELECT * FROM superstore LIMIT 20):", height=150)
#     if st.button("Run SQL"):
#         if not sql.strip():
#             st.warning("Enter SQL.")
#         else:
#             try:
#                 with engine.connect() as conn:
#                     df = pd.read_sql(sql, conn)
#                 st.dataframe(df)
#                 st.success(f"Returned {len(df)} rows.")
#             except Exception as e:
#                 st.error(f"SQL execution failed: {e}\n\n{traceback.format_exc()}")

# # --- Table Comparison ---
# elif mode == "Table Comparison":
#     st.header("📋 Table Comparison (table_agent)")
#     st.markdown("Ask natural-language comparisons: e.g. 'compare top products', 'sales by category', 'compare product A and B'.")
#     t_q = st.text_input("Comparison instruction:", key="table_input")
#     if st.button("Run comparison"):
#         if not t_q.strip():
#             st.warning("Enter a comparison query.")
#         else:
#             try:
#                 out = table_agent.table_agent(t_q)
#                 # table_agent returns markdown string; show it in code block for readability
#                 st.code(out)
#             except Exception as e:
#                 st.error(f"Table agent error: {e}\n\n{traceback.format_exc()}")

# # --- Memory ---
# elif mode == "Memory":
#     st.header("🧠 Short-term Memory")
#     mem = agent.memory if hasattr(agent, "memory") else None
#     if not mem:
#         st.info("No ShortTermMemory found in agentic_system.py")
#     else:
#         st.write(f"Capacity: {mem.max} items. Stored: {len(mem.items)} items.")
#         if len(mem.items) == 0:
#             st.info("Memory empty.")
#         else:
#             for ts, query, agent_name, answer in reversed(mem.items):
#                 st.markdown(f"- **{pd.to_datetime(ts, unit='s')}** — *{query}* → ({agent_name})")
#                 # show short snippet of answer
#                 snippet = (answer[:400] + "...") if isinstance(answer, str) and len(answer) > 400 else answer
#                 st.write(snippet)

#         if st.button("Clear memory"):
#             mem.items = []
#             st.success("Memory cleared.")

# # Footer / notes
# st.markdown("---")
# st.caption("Notes: This dashboard uses your local modules (agentic_system, analytics_visuals, table_agent). Ensure your .env is set and PostgreSQL + Pinecone (if used) are reachable.")



# # app.py
# import streamlit as st
# import traceback
# import pandas as pd
# import os
# import base64

# # import your backend modules
# import agentic_system as agent
# import table_agent
# import analytics_visuals
# from db_connection import engine  # optional: for raw SQL execution

# st.set_page_config(page_title="Agentic AI — Core Dashboard", layout="wide")
# st.title("🤖 AI Knowledge Assistant")
# st.markdown("Chat + Visuals + Table Comparison + Short-term Memory, with inline chart & PDF display.")

# # Session state
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Sidebar
# st.sidebar.header("Navigation")
# mode = st.sidebar.radio("Choose page", ["Chat", "Visualization", "SQL Runner", "Table Comparison", "Memory"])

# st.sidebar.markdown("---")
# st.sidebar.write("Env checks:")
# try:
#     num_tables = len(agent.public_tables) if hasattr(agent, "public_tables") else "N/A"
# except Exception:
#     num_tables = "N/A"
# st.sidebar.write(f"Postgres tables: {num_tables}")
# pc_status = "configured" if (hasattr(agent, "PINECONE_API_KEY") and agent.PINECONE_API_KEY) else "not configured"
# st.sidebar.write(f"Pinecone: {pc_status}")

# # Helper to render chat history
# def render_history():
#     if not st.session_state.history:
#         st.info("No chat history yet.")
#         return
#     for role, text in reversed(st.session_state.history):
#         if role == "You":
#             st.markdown(f"**🧑‍💻 You:** {text}")
#         else:
#             st.markdown(f"**🤖 Bot:** {text}")

# # Helper to display PDF inline
# def display_pdf(pdf_path):
#     if not os.path.exists(pdf_path):
#         st.error(f"PDF not found: {pdf_path}")
#         return
    
#     with open(pdf_path, "rb") as f:
#         pdf_bytes = f.read()

#     b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

#     st.download_button(
#         label="📄 Download PDF",
#         data=pdf_bytes,
#         file_name=os.path.basename(pdf_path),
#         mime="application/pdf"
#     )

#     st.markdown(
#         f"""
#         <iframe src="data:application/pdf;base64,{b64_pdf}"
#         width="800" height="600" type="application/pdf"></iframe>
#         """,
#         unsafe_allow_html=True
#     )

# # Helper: display chart inline
# def display_chart(file_path):
#     if os.path.exists(file_path) and file_path.lower().endswith((".png", ".jpg", ".jpeg")):
#         st.image(file_path, caption="Generated Chart", use_column_width=True)
#         return True
#     return False


# # ==============================
# # CHAT
# # ==============================
# if mode == "Chat":
#     st.header("💬 Chat (RAG / SQL / ML / Report / Table / Visualization)")

#     q = st.text_input("Ask your question:", key="chat_input")

#     if st.button("Send"):
#         if not q.strip():
#             st.warning("Please type a question.")
#         else:
#             try:
#                 label, resp = agent.route_query(q)
#             except Exception as e:
#                 label = "ERROR"
#                 resp = f"Exception calling route_query:\n{e}\n\n{traceback.format_exc()}"

#             st.markdown(f"**[{label}]**")
#             st.session_state.history.append(("You", q))

#             # -----------------------------
#             # PDF DETECTION FIX (NEW CODE)
#             # -----------------------------

#             pdf_path = None

#             if isinstance(resp, str):
#                 # Case 1: response IS the PDF path
#                 if resp.strip().endswith(".pdf") and os.path.exists(resp.strip()):
#                     pdf_path = resp.strip()
#                 else:
#                     # Case 2: extract PDF path from inside response text
#                     for word in resp.split():
#                         if word.endswith(".pdf") and os.path.exists(word):
#                             pdf_path = word
#                             break

#             if pdf_path:
#                 st.success("📄 PDF Report Generated!")
#                 display_pdf(pdf_path)
#                 st.session_state.history.append(("Bot", f"[PDF Generated] {pdf_path}"))
#             else:
#                 # If not a PDF, maybe it's a chart
#                 shown_chart = False
#                 if isinstance(resp, str):
#                     for word in resp.split():
#                         if display_chart(word):
#                             shown_chart = True
#                             break

#                 # Default text output
#                 if not shown_chart:
#                     if isinstance(resp, str) and ("\n" in resp or len(resp) > 400):
#                         st.code(resp)
#                     else:
#                         st.write(resp)

#                 st.session_state.history.append(("Bot", resp))

#     st.markdown("---")
#     st.subheader("Chat History")
#     render_history()


# # ==============================
# # VISUALIZATION
# # ==============================
# elif mode == "Visualization":
#     st.header("📈 Visualization (auto-detect)")
#     vis_q = st.text_input("Visualization instruction:")

#     if st.button("Generate visualization"):
#         if not vis_q.strip():
#             st.warning("Enter an instruction.")
#         else:
#             try:
#                 out = agent.visualization_agent(vis_q)
#                 st.success("Visualization agent response:")
#                 st.write(out)

#                 # Try showing chart directly
#                 if isinstance(out, str):
#                     if not display_chart(out):
#                         for word in out.split():
#                             if display_chart(word):
#                                 break

#             except Exception as e:
#                 st.error(f"Visualization agent error:\n{e}\n\n{traceback.format_exc()}")

#     st.markdown("---")
#     st.write("Recent charts:")
#     visuals_dir = "reports/visuals"
#     if os.path.isdir(visuals_dir):
#         images = [f for f in os.listdir(visuals_dir) if f.endswith((".png", ".jpg"))]
#         for img in sorted(images, reverse=True)[:3]:
#             st.image(os.path.join(visuals_dir, img), caption=img, use_column_width=True)
#     else:
#         st.info("No charts yet.")


# # ==============================
# # SQL RUNNER
# # ==============================
# elif mode == "SQL Runner":
#     st.header("🧾 SQL Runner")
#     sql = st.text_area("Enter SQL:")

#     if st.button("Run SQL"):
#         if not sql.strip():
#             st.warning("Enter SQL.")
#         else:
#             try:
#                 with engine.connect() as conn:
#                     df = pd.read_sql(sql, conn)
#                 st.dataframe(df)
#             except Exception as e:
#                 st.error(f"SQL error:\n{e}\n\n{traceback.format_exc()}")


# # ==============================
# # TABLE COMPARISON
# # ==============================
# elif mode == "Table Comparison":
#     st.header("📋 Table Comparison")
#     t_q = st.text_input("Comparison instruction:")

#     if st.button("Run comparison"):
#         try:
#             out = table_agent.table_agent(t_q)
#             st.code(out)
#         except Exception as e:
#             st.error(f"Table agent error:\n{e}\n\n{traceback.format_exc()}")


# # ==============================
# # MEMORY
# # ==============================
# elif mode == "Memory":
#     st.header("🧠 Short-term Memory")
#     mem = agent.memory if hasattr(agent, "memory") else None

#     if not mem:
#         st.info("No memory found.")
#     else:
#         st.write(f"Stored items: {len(mem.items)}")
#         for ts, q, a, ans in reversed(mem.items):
#             st.markdown(f"- **{pd.to_datetime(ts, unit='s')}** — *{q}* → ({a})")
#             st.write(str(ans)[:400] + ("..." if len(str(ans)) > 400 else ""))

#         if st.button("Clear memory"):
#             mem.items = []
#             st.success("Memory cleared.")

# # Footer
# st.markdown("---")
# st.caption("Enhanced frontend: Inline charts + PDF viewer + PDF detection fix + full agent routing.")




# app.py
import streamlit as st
import traceback
import pandas as pd
import os
import base64

# backend modules
import agentic_system as agent
import table_agent
import analytics_visuals

st.set_page_config(page_title="Agentic AI — Chat Only", layout="wide")
st.title("🤖 Agentic AI — Unified Chat Interface")
st.markdown("Ask anything: SQL • RAG • ML Forecast • Reports • Visuals • Comparison Tables")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Helper: Display PDF inline
def display_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found: {pdf_path}")
        return
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=os.path.basename(pdf_path))
    st.markdown(
        f"""
        <iframe src="data:application/pdf;base64,{b64_pdf}" 
        width="800" height="600" type="application/pdf"></iframe>
        """,
        unsafe_allow_html=True
    )

# Helper: Display chart inline
def display_chart(path):
    if os.path.exists(path) and path.lower().endswith((".png", ".jpg", ".jpeg")):
        st.image(path, caption="Generated Chart", use_column_width=True)
        return True
    return False



# ===========================
#            CHAT
# ===========================
st.header("💬 Chat")

query = st.text_input("Type your question:", key="chat_input")

if st.button("Send"):
    if not query.strip():
        st.warning("Please type something.")
    else:
        try:
            label, resp = agent.route_query(query)
        except Exception as e:
            label = "ERROR"
            resp = f"Exception in router:\n{e}\n\n{traceback.format_exc()}"

        # store user message
        st.session_state.history.append(("You", query))
        st.markdown(f"**[{label}]**")

        # ----- PDF DETECTION -----
        pdf_path = None
        if isinstance(resp, str):
            # case 1: response IS pdf
            if resp.strip().endswith(".pdf") and os.path.exists(resp.strip()):
                pdf_path = resp.strip()
            else:
                # case 2: extract pdf path from text
                for word in resp.split():
                    if word.endswith(".pdf") and os.path.exists(word):
                        pdf_path = word
                        break

        if pdf_path:
            st.success("📄 PDF Generated:")
            display_pdf(pdf_path)
            st.session_state.history.append(("Bot", f"[PDF Generated] {pdf_path}"))
        else:
            # ----- CHART DETECTION -----
            shown_chart = False
            if isinstance(resp, str):
                for word in resp.split():
                    if display_chart(word):
                        shown_chart = True
                        break

            if not shown_chart:
                # default text output
                if isinstance(resp, str) and ("\n" in resp or len(resp) > 400):
                    st.code(resp)
                else:
                    st.write(resp)

            st.session_state.history.append(("Bot", resp))



# ===========================
#     CHAT HISTORY
# ===========================
st.markdown("---")
st.subheader("🕘 Conversation History")

for role, msg in reversed(st.session_state.history):
    if role == "You":
        st.markdown(f"**🧑‍💻 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

st.markdown("---")
st.caption("Unified Chat — All agents (SQL, RAG, ML, Reports, Visualization, Table) handled inside chat.")
