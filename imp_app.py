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

    prompt = f"""
    You are a data assistant analyzing CO₂ emissions dataset.
    Context from retrieved records:
    {context}

    User question: {query}

    Instructions:
    - Summarize all retrieved activities and category summaries in a concise, data-driven manner.
    - Provide 2–3 key insights across different categories or activities.
    - Suggest clear, actionable sustainability improvements.
    - Estimate a realistic % reduction in emissions if the suggested actions are followed.
    - Respond strictly in this format:
      Answer: ...
      Suggestions: ...
      Estimated Reduction: X% ...
      Long-term: ...
    """

    response = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    output = response.choices[0].message.content.strip()

    reduction_match = re.search(r"(\d+)%", output)
    reduction = float(reduction_match.group(1)) if reduction_match else 10.0

    suggestions_match = re.search(r"Suggestions:(.*?)(?:Estimated|$)", output, re.S)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions found."

    return output, context, confidence, metadatas, reduction, suggestions


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🌍 CO₂ Insights RAG", page_icon="🌱", layout="wide")
st.title("🌍 RAG-powered CO₂ Emission Insights")
st.caption("Smart analysis using ChromaDB + Groq + Streamlit")

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

    st.markdown(f"**Confidence Score:** {confidence:.1f}%")

    with st.expander("📚 Retrieved Context"):
        st.write(context)

    st.markdown("### 🌱 Sustainability Suggestions")
    st.info(suggestions)

    # --- Visualization ---
    if metadatas:
        st.subheader("📊 CO₂ Emissions — Before vs After Recommendations")
        retrieved_df = pd.DataFrame(metadatas)
        top_activities = retrieved_df.nlargest(5, "emission").copy()
        top_activities["Reduced Emission"] = top_activities["emission"] * (1 - reduction / 100)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(top_activities["activity"], top_activities["emission"], color="#FF6B6B", label="Before")
        ax.bar(top_activities["activity"], top_activities["Reduced Emission"], color="#51CF66", label=f"After ({reduction:.1f}%↓)")
        ax.set_ylabel("CO₂ Emission (kg/day)")
        ax.set_title("Before vs After Sustainability Actions", fontsize=12, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        # Show numeric comparison
        st.markdown("### 📈 Emission Comparison Table")
        st.dataframe(
            top_activities[["activity", "emission", "Reduced Emission", "category"]]
            .rename(columns={"emission": "Before (kg/day)", "Reduced Emission": "After (kg/day)"})
            .reset_index(drop=True)
        )

st.markdown("---")
st.caption("Built with ❤️ using Groq, ChromaDB, and Streamlit.")





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
    q_emb = embedder.encode([query])[0].tolist()
    filters = {}
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

    prompt = f"""
     You are a data assistant analyzing CO₂ emissions dataset.
     Context from retrieved records:
     {context}

     User question: {query}

     Instructions:
     - Summarize all retrieved activities and category summaries in a concise, data-driven manner.
     - Provide 2–3 key insights across different categories or activities.
     - Suggest clear, actionable sustainability improvements.
     - Estimate a realistic % reduction in emissions if the suggested actions are followed.
     - Respond strictly in the following format:

     Respond in this format: 
     Answer: ... 
     Suggestions: ... 
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

    reduction_match = re.search(r"(\d+)%", output)
    reduction = float(reduction_match.group(1)) if reduction_match else 10.0
    suggestions_match = re.search(r"Suggestions:(.*?)(?:Estimated|$)", output, re.S)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions found."

    return output, context, confidence, metadatas, reduction, suggestions

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🌍 CO₂ Insights RAG", page_icon="🌱", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
    }
    .stMarkdown, .stTextInput, .stDataFrame {
        text-align: center;
    }
    h1, h2, h3, h4 {
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #00A86B;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #008a5a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 RAG-powered CO₂ Insights with LLM")
st.caption("<p style='text-align:center;'>Upload your CO₂ dataset and get data-driven sustainability recommendations.</p>", unsafe_allow_html=True)

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

    st.markdown("<h3 style='text-align:center;'>🤖 AI-Powered Answer</h3>", unsafe_allow_html=True)
    st.success(answer)

    st.markdown(f"<p style='text-align:center;'><b>Confidence Score:</b> {confidence:.2f}%</p>", unsafe_allow_html=True)

    with st.expander("📚 Retrieved Context"):
        st.write(context)

    st.markdown("<h3 style='text-align:center;'>🌱 Sustainability Suggestions</h3>", unsafe_allow_html=True)
    st.info(suggestions)

    # --- Visualization ---
    if metadatas:
        st.markdown("<h3 style='text-align:center;'>📊 CO₂ Emissions Overview</h3>", unsafe_allow_html=True)
        retrieved_df = pd.DataFrame(metadatas)
        top_activities = retrieved_df.nlargest(5, "emission").copy()
        top_activities["Reduced Emission"] = top_activities["emission"] * (1 - reduction / 100)

        fig, ax = plt.subplots(figsize=(6, 4))
        bar_width = 0.35
        x = range(len(top_activities))

        ax.bar(x, top_activities["emission"], width=bar_width, color="#FF6B6B", label="Before")
        ax.bar([p + bar_width for p in x], top_activities["Reduced Emission"], width=bar_width, color="#51CF66", label=f"After ({reduction:.1f}%↓)")

        ax.set_ylabel("CO₂ Emission (kg/day)", fontsize=10)
        ax.set_title("Before vs After Sustainability Actions", fontsize=12, fontweight="bold")
        ax.set_xticks([p + bar_width / 2 for p in x])
        ax.set_xticklabels(top_activities["activity"], rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        # --- Comparison Table ---
        styled_df = (
            top_activities[["activity", "emission", "Reduced Emission", "category"]]
            .rename(columns={
                "activity": "Activity",
                "emission": "Before (kg/day)",
                "Reduced Emission": "After (kg/day)",
                "category": "Category"
            })
            .reset_index(drop=True)
        )

        st.markdown("<h4 style='text-align:center;'>📈 Emission Comparison Table</h4>", unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True)

st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:gray; font-size:12px;'>
    Made with 💚 using <b>Streamlit + LLM + ChromaDB</b> | CO₂ RAG Bot © 2025
    </div>
""", unsafe_allow_html=True)
