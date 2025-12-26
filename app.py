import os
import uuid
import hashlib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
from groq import Groq
import re

# ---------------- CONFIG ----------------
CSV_FILE = "CO2_Emission_Dataset_200.csv"
COLUMNS = ["Activity", "Avg_CO2_Emission(kg/day)", "Category"]
CHROMA_PERSIST_DIR = "./chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 50
CHUNK_SIZE = 300
# ----------------------------------------

# Load env vars
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

# Configure Groq client
client_groq = Groq(api_key=API_KEY)

# Init Chroma + Embedder
embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection("docs")

# ---- Utils ----
def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def index_csv(file_path):
    file_hash = get_file_hash(file_path)
    existing = collection.get(include=["metadatas"])

    if existing["metadatas"] and any(
        m and m.get("file_hash") == file_hash for m in existing["metadatas"]
    ):
        return False

    df = pd.read_csv(file_path)
    ids, docs, metas = [], [], []

    # --- Row-level indexing ---
    for i, row in df.iterrows():
        doc_id = str(i)
        text = (
            f"Activity: {row['Activity']} | "
            f"Emission: {row['Avg_CO2_Emission(kg/day)']} kg/day | "
            f"Category: {row['Category']}"
        )
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{doc_id}_{idx}")
            docs.append(chunk)
            metas.append({
                "file_hash": file_hash,
                "activity": row["Activity"],
                "emission": row["Avg_CO2_Emission(kg/day)"],
                "category": row["Category"],
                "type": "row"
            })

    # --- Category-level summaries ---
    category_summary = (
        df.groupby("Category")["Avg_CO2_Emission(kg/day)"]
        .mean()
        .reset_index()
    )
    for _, row in category_summary.iterrows():
        doc_id = str(uuid.uuid4())
        text = f"Category Summary: {row['Category']} has an average emission of {row['Avg_CO2_Emission(kg/day)']:.2f} kg/day."
        ids.append(doc_id)
        docs.append(text)
        metas.append({
            "file_hash": file_hash,
            "activity": f"{row['Category']} summary",
            "emission": row["Avg_CO2_Emission(kg/day)"],
            "category": row["Category"],
            "type": "category_summary"
        })

    # --- Upsert ---
    embeddings = embedder.encode(docs, convert_to_numpy=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return True

def hybrid_retrieval(query, top_k=TOP_K):
    """Semantic search + optional keyword filter if category mentioned."""
    q_emb = embedder.encode([query])[0].tolist()
    filters = {}

    # Basic keyword filters
    keywords = ["Transport", "Household", "Food", "Energy"]
    for kw in keywords:
        if kw.lower() in query.lower():
            filters["category"] = kw
            break

    if filters:
        results = collection.query(query_embeddings=[q_emb], n_results=top_k, where=filters)
    else:
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    return results

def ask(query, top_k=TOP_K):
    results = hybrid_retrieval(query, top_k)

    if not results["documents"]:
        return "No documents found.", "", 0.0, [], 0.0, ""

    context_docs = results["documents"][0]
    context = "\n\n".join(context_docs)
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    similarities = [(1 - distance / 2) * 100 for distance in distances]
    confidence = round(sum(similarities) / len(similarities), 2)

    # Improved prompt
    prompt = f"""
    You are a data assistant analyzing CO₂ emissions dataset.
    Context from retrieved records:
    {context}

    User question: {query}

    Instructions:
    - Summarize all retrieved activities and category summaries in a concise, data-driven manner.
    - Provide 2–3 key insights across different categories or activities.
    - Suggest clear, actionable sustainability improvements.
    - List each suggestion on a separate line, starting with a dash (-).
    - Estimate a realistic % reduction in emissions if the suggested actions are followed.
    - Respond strictly in the following format:

    Respond in this format: 
    Answer: ... 
    Suggestions:
    - ...
    - ...
    - ...
    Estimated Reduction: X% ...
    Long-term: [Sustainable improvement or transition advice]...
    Be factual, concise, and avoid speculation.
    """

    response = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    output = response.choices[0].message.content.strip()

    # Extract reduction %
    reduction_match = re.search(r"(\d+)%", output)
    reduction = float(reduction_match.group(1)) if reduction_match else 10.0

    # Extract suggestions
    suggestions_match = re.search(r"Suggestions:(.*?)(?:Estimated|$)", output, re.S)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions found."

    return output, context, confidence, metadatas, reduction, suggestions

# ---------------- Streamlit UI ----------------
# Apply custom CSS for center theme and professional look
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .block-container {
        max-width: 1200px;
        margin: auto;
        padding-top: 30px;
    }
    h1, h2, h3, h4 {
        text-align: center !important;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #222;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-size: 16px;
        background-color: #0078d4;
        color: white;
        border: none;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .stTextInput>div>div>input {
        text-align: left;
        font-size: 16px;
        border-radius: 8px;
    }
    .stMarkdown, .stCaption, .stSuccess, .stInfo {
        text-align: center;
    }
    .stExpander {
        border-radius: 8px;
        background-color: #e9ecef;
    }
    .stSubheader {
        text-align: center;
    }
    .stPlotlyChart, .stPyplot {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="CO₂ RAG with Groq", page_icon="🌍", layout="wide")

st.title("🌍 RAG-powered CO₂ Insights with llm")
st.caption("Upload a CO₂ dataset and ask questions with retrieval-augmented insights.")

# Sidebar
st.sidebar.header("⚙️ Settings")
if st.sidebar.button("Re-index CSV"):
    collection.delete(where={})
    indexed = index_csv(CSV_FILE)
    st.sidebar.success("✅ Re-indexed successfully!" if indexed else "ℹ️ No changes detected.")

indexed = index_csv(CSV_FILE)
if indexed:
    st.sidebar.success("✅ Indexed CSV")
else:
    st.sidebar.info("ℹ️ CSV already indexed")

# Query Section
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("🔍 Searching and generating answer..."):
        answer, context, confidence, metadatas, reduction, suggestions = ask(query)

    st.markdown("### 🤖 Answer")
    st.success(answer)

    if confidence >= 80:
        st.markdown(f"✅ **Confidence Score:** {confidence}% (High)")
    elif confidence >= 50:
        st.markdown(f"🟠 **Confidence Score:** {confidence}% (Medium)")
    else:
        st.markdown(f"🔴 **Confidence Score:** {confidence}% (Low)")

    with st.expander("📚 Retrieved Context"):
        st.write(context)

    st.markdown("### 🌱 Model’s Sustainability Suggestions")
    st.info(suggestions)

    # --- Visualization ---
    if metadatas:
        st.subheader("📊 CO₂ Emissions Overview")
        retrieved_df = pd.DataFrame(metadatas)
        top_activities = retrieved_df.nlargest(5, "emission")

        col1, col2 = st.columns(2)

        # --- Current Emissions ---
        with col1:
            st.markdown("**Current Emissions (Top 5)**")
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(
                top_activities["activity"],
                top_activities["emission"],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                color="blue",
                label="Current"
            )
            ax.set_ylabel("kg/day", fontsize=9)
            ax.set_title("Current CO₂", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax.legend(fontsize=8)
            st.pyplot(fig)

        # --- After Suggestions ---
        with col2:
            st.markdown("**Predicted Reduction (Top 5)**")
            reduced_df = top_activities.copy()
            reduced_df["Reduced Emission"] = reduced_df["emission"] * (1 - reduction / 100)

            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.plot(
                reduced_df["activity"], 
                reduced_df["emission"], 
                marker="o", 
                linestyle="-", 
                color="red", 
                label="Before"
            )
            ax2.plot(
                reduced_df["activity"], 
                reduced_df["Reduced Emission"], 
                marker="s", 
                linestyle="--", 
                color="green", 
                label=f"After ({reduction:.1f}%↓)"
            )
            ax2.set_ylabel("kg/day", fontsize=9)
            ax2.set_title("After Suggestions", fontsize=11)
            ax2.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax2.legend(fontsize=8)
            st.pyplot(fig2)

st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:gray; font-size:12px;'>
    Made using <b>Streamlit + llm + ChromaDB</b> | CO₂ RAG Bot © 2025
    </div>
    """, unsafe_allow_html=True)

