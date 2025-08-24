import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ------------------------------
# Loader
# ------------------------------
@st.cache_data
def load_bioasq(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data.get("data", []):
        for para in item.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                questions.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "answers": [a["text"] for a in qa.get("answers", [])],
                    "context": context
                })
    return questions

# ------------------------------
# Analysis Functions
# ------------------------------
def get_stats(questions):
    num_qas = len(questions)
    num_unique_contexts = len(set(q["context"] for q in questions))
    avg_context_len = sum(len(q["context"].split()) for q in questions) / num_qas
    avg_question_len = sum(len(q["question"].split()) for q in questions) / num_qas
    avg_answer_len = sum(len(a.split()) for q in questions for a in q["answers"]) / num_qas

    return {
        "Total QAs": num_qas,
        "Unique Contexts": num_unique_contexts,
        "Avg Context Length (words)": round(avg_context_len, 2),
        "Avg Question Length (words)": round(avg_question_len, 2),
        "Avg Answer Length (words)": round(avg_answer_len, 2),
    }

def question_type_distribution(questions):
    wh_words = ["what", "which", "when", "where", "who", "why", "how", "name"]
    types = []
    for q in questions:
        first = q["question"].strip().lower().split()[0]
        if first in wh_words:
            types.append(first)
        else:
            types.append("other")
    return Counter(types)

def answer_length_distribution(questions):
    return [len(a.split()) for q in questions for a in q["answers"]]

def context_length_distribution(questions):
    return [len(q["context"].split()) for q in questions]

# -------------------------
# PubMedQA Analysis Section
# -------------------------
def pubmedqa_analysis(df_bioasq):
    st.header("📊 PubMedQA Dataset Analysis")

    PUBMEDQA_PATH = "ori_pqal.json"   # adjust path as needed

    def load_pubmedqa(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    data = load_pubmedqa(PUBMEDQA_PATH)

    # Convert to DataFrame
    records = []
    for pmid, sample in data.items():
        records.append({
            "PMID": pmid,
            "Question": sample.get("QUESTION", ""),
            "Contexts": sample.get("CONTEXTS", []),
            "Answer": sample.get("final_decision", ""),
            "Reasoning Required": sample.get("reasoning_required_pred", "unknown"),
            "Reasoning Free": sample.get("reasoning_free_pred", "unknown"),
            "Year": sample.get("YEAR", "unknown"),
            "MeSH": sample.get("MESHES", [])
        })

    df_pubmedqa = pd.DataFrame(records)

    st.write("### Dataset Preview")
    st.dataframe(df_pubmedqa.head(10))

    # 1. Distribution of answer types
    st.subheader("Answer Type Distribution")
    fig, ax = plt.subplots()
    df_pubmedqa["Answer"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Answers (Yes/No/Maybe)")
    st.pyplot(fig)

    # 2. Year distribution
    st.subheader("Publication Year Distribution")
    fig, ax = plt.subplots()
    df_pubmedqa["Year"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Distribution by Publication Year")
    st.pyplot(fig)

    # 3. Reasoning requirement
    st.subheader("Reasoning Requirement Analysis")
    reasoning_counts = df_pubmedqa["Reasoning Required"].value_counts()
    fig, ax = plt.subplots()
    reasoning_counts.plot(kind="bar", ax=ax)
    ax.set_title("Reasoning Required vs Not")
    st.pyplot(fig)

   

    # 5. Context length distribution
    st.subheader("Context Length Distribution")
    fig, ax = plt.subplots()
    df_pubmedqa["Contexts"].apply(len).plot(kind="hist", bins=20, ax=ax)
    ax.set_title("Histogram of Context Counts per Question")
    st.pyplot(fig)

    # ------------------------------
    # ⚖️ BioASQ vs PubMedQA Comparison
    # ------------------------------
    st.header("⚖️ BioASQ vs PubMedQA: Comparative Benchmarking")

    # Prepare comparison metrics
    comparison_data = {
        "Dataset": ["BioASQ", "PubMedQA"],
        "Total Questions": [len(df_bioasq), len(df_pubmedqa)],
        "Avg Context Length": [
            df_bioasq["context"].apply(lambda x: len(x.split())).mean(),
            df_pubmedqa["Contexts"].apply(lambda x: sum(len(c.split()) for c in x)/len(x) if len(x)>0 else 0).mean()
        ],
        "Avg Answer Length": [
            df_bioasq["answers"].apply(lambda x: len(x[0].split()) if isinstance(x, list) and len(x)>0 else 0).mean(),
            0  # PubMedQA doesn't have free-text answers, just Yes/No/Maybe
        ],
        "Question Type Diversity": ["Factoid/List/Yes-No/Summary", "Yes/No/Maybe"]
    }

    comp_df = pd.DataFrame(comparison_data)

    # 📑 Show comparison table
    st.subheader("📑 Benchmark Comparison Table")
    st.table(comp_df)

    # 📊 Bar chart for key metrics
    st.subheader("📊 Key Metric Comparison")
    metrics_df = comp_df.melt(
        id_vars="Dataset", 
        value_vars=["Total Questions", "Avg Context Length", "Avg Answer Length"],
        var_name="Metric", 
        value_name="Value"
    )

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Metric", y="Value", hue="Dataset", data=metrics_df, ax=ax)
    ax.set_title("BioASQ vs PubMedQA Metrics")
    st.pyplot(fig)

    # 🥧 Pie chart: Question type distribution
    st.subheader("🥧 Question Type Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BioASQ Question Types**")
        if "type" in df_bioasq.columns:
            bioasq_qtypes = df_bioasq["type"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(bioasq_qtypes, labels=bioasq_qtypes.index, autopct="%1.1f%%", startangle=140)
            ax1.axis("equal")
            st.pyplot(fig1)
        else:
            st.info("BioASQ dataset here does not include explicit 'type' field.")

    with col2:
        st.markdown("**PubMedQA Question Labels (Final Decision)**")
        pubmedqa_qtypes = df_pubmedqa["Answer"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(pubmedqa_qtypes, labels=pubmedqa_qtypes.index, autopct="%1.1f%%", startangle=140)
        ax2.axis("equal")
        st.pyplot(fig2)

    # ✅ Final justification
    st.success("✅ BioASQ provides richer, more diverse, and realistic benchmarking for PubMed retrieval compared to PubMedQA, which is limited to Yes/No/Maybe answers.")

    st.success("✅ PubMedQA Analysis Completed")

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.title("📊 BioASQ & PubMedQA Dataset Analysis for PubMed Retrieval")

    # ----------------- BioASQ -----------------
    path = "BioASQ-train-factoid-6b-full-annotated.json"
    questions = load_bioasq(path)
    df = pd.DataFrame(questions)

    st.subheader("📌 BioASQ Dataset Statistics")
    stats = get_stats(questions)
    st.table(pd.DataFrame(stats.items(), columns=["Metric", "Value"]))

    st.subheader("🔎 Question Type Distribution")
    qtypes = question_type_distribution(questions)
    qtype_df = pd.DataFrame(qtypes.items(), columns=["Type", "Count"])
    st.bar_chart(qtype_df.set_index("Type"))

    st.subheader("📝 Answer Length Distribution")
    lengths = answer_length_distribution(questions)
    fig, ax = plt.subplots()
    sns.histplot(lengths, bins=20, ax=ax)
    ax.set_xlabel("Answer Length (words)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("📖 Context Length Distribution")
    clen = context_length_distribution(questions)
    fig, ax = plt.subplots()
    sns.histplot(clen, bins=30, ax=ax)
    ax.set_xlabel("Context Length (words)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("📍 Sample Questions & Answers")
    st.write(df.sample(5))

    st.subheader("📑 Why BioASQ for Benchmarking?")
    justification = pd.DataFrame([
        ["Domain-specific (biomedical)", "Covers PubMed-like biomedical questions ensuring relevance"],
        ["Large & diverse", "Covers factoid, list, yes/no, and summary questions"],
        ["Real-world complexity", "Contexts from PubMed abstracts simulate true search/retrieval needs"],
        ["Standard benchmark", "Widely used in biomedical IR & QA competitions"],
        ["Supports multiple tasks", "Useful for retrieval, re-ranking, and answer generation evaluation"]
    ], columns=["Reason", "Justification"])
    st.table(justification)

    # ----------------- PubMedQA -----------------
    pubmedqa_analysis(df)

if __name__ == "__main__":
    main()
