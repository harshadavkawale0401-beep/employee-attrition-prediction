# ===============================
# Fake Job Posting Detection
# Streamlit Dashboard
# ===============================

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Fake Job Detector", layout="wide")
st.title(" Fake Job Posting Detector")

# -------------------------------
# Load models & data
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_fake_job_model.pkl")

@st.cache_resource
def load_clustering_pipeline():
    return joblib.load("job_clustering_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("fake_job_postings.csv")

try:
    model_pkg = load_model()
    cluster_pkg = load_clustering_pipeline()
    df = load_data()
except Exception as e:
    st.error("Error loading models or data. Ensure all files are in the same folder as app.py.")
    st.stop()

best_model = model_pkg["model"]
tfidf = model_pkg["tfidf"]
ohe = model_pkg["ohe"]
cat_cols = model_pkg["cat_cols"]
best_name = model_pkg["best_name"]

tfidf_cluster = cluster_pkg["tfidf_cluster"]
svd = cluster_pkg["svd"]
kmeans = cluster_pkg["kmeans"]
best_k = cluster_pkg["best_k"]

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Model Performance",
    "Clustering Insights",
    "Predict New Job"
])

# ===============================
# Tab 1: Dataset Overview
# ===============================
with tab1:
    st.header("Dataset Overview")
    st.write(f"Total job postings: **{len(df)}**")
    fraud_counts = df["fraudulent"].value_counts().reset_index()
    fraud_counts.columns = ["fraudulent", "Count"]
    fraud_counts["Label"] = fraud_counts["fraudulent"].map({0: "Real", 1: "Fake"})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fake vs Real")
        fig = px.bar(
            fraud_counts,
            x="Label",
            y="Count",
            color="Label",
            title="Fake vs Real Job Postings"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraudulent %")
        fake_pct = fraud_counts[fraud_counts["fraudulent"] == 1]["Count"].sum() / len(df) * 100
        st.metric("Percentage of Fake Jobs", f"{fake_pct:.2f}%")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Top Industries")
    top_ind = df["industry"].value_counts().head(10).reset_index()
    top_ind.columns = ["Industry", "Count"]
    fig_ind = px.bar(
        top_ind,
        x="Count",
        y="Industry",
        orientation="h",
        title="Top 10 Industries"
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    st.subheader("Top Functions")
    top_fun = df["function"].value_counts().head(10).reset_index()
    top_fun.columns = ["Function", "Count"]
    fig_fun = px.bar(
        top_fun,
        x="Count",
        y="Function",
        orientation="h",
        title="Top 10 Functions"
    )
    st.plotly_chart(fig_fun, use_container_width=True)

# ===============================
# Tab 2: Model Performance
# ===============================
with tab2:
    st.header("Model Performance")
    st.write(f"Best model: **{best_name}** (chosen by F1 on fake class).")

    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("")
    )
    def clean_text(t):
        t = str(t).lower()
        t = " ".join(t.split())
        return t
    df["text_clean"] = df["text"].apply(clean_text)

    X_tfidf = tfidf.transform(df["text_clean"])
    X_cat = ohe.transform(df[cat_cols].fillna("Unknown"))
    X = np.hstack([X_tfidf.toarray(), X_cat])
    y = df["fraudulent"].values

    y_pred = best_model.predict(X)

    st.subheader("Classification Report (full data)")
    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True, target_names=["Real", "Fake"])).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=["Real", "Fake"],
        y=["Real", "Fake"],
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ===============================
# Tab 3: Clustering Insights
# ===============================
with tab3:
    st.header("Job Clustering Insights")
    st.write(f"Number of clusters: **{best_k}**")

    df["desc_text"] = df["description"].fillna("").apply(lambda t: " ".join(str(t).lower().split()))
    X_desc = tfidf_cluster.transform(df["desc_text"])
    X_red = svd.transform(X_desc)
    clusters = kmeans.predict(X_red)
    df["cluster"] = clusters

    st.subheader("Fake Ratio per Cluster")
    cluster_stats = df.groupby("cluster")["fraudulent"].mean().reset_index()
    cluster_stats.columns = ["Cluster", "Fake Ratio"]
    fig_clust = px.bar(
        cluster_stats,
        x="Cluster",
        y="Fake Ratio",
        title="Proportion of Fake Jobs per Cluster"
    )
    st.plotly_chart(fig_clust, use_container_width=True)

    st.subheader("Cluster Details")
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        fake_ratio = sub["fraudulent"].mean()
        top_ind = sub["industry"].value_counts().head(3)
        top_fun = sub["function"].value_counts().head(3)

        st.write(f"**Cluster {c}** – Fake ratio: {fake_ratio:.2f}")
        if not top_ind.empty:
            st.write("Top Industries:")
            st.write(top_ind)
        if not top_fun.empty:
            st.write("Top Functions:")
            st.write(top_fun)

        st.write("Sample jobs:")
        samples = sub[["title", "description", "fraudulent"]].sample(min(5, len(sub)), random_state=42)
        st.dataframe(samples, use_container_width=True)
        st.divider()

# ===============================
# Tab 4: Predict New Job
# ===============================
with tab4:
    st.header("Predict New Job Posting")
    st.write("Fill in details to check if a job is likely fake or real.")

    title = st.text_area("Job Title", height=50)
    description = st.text_area("Job Description", height=150)
    requirements = st.text_area("Requirements", height=100)

    emp_types = sorted(df["employment_type"].dropna().unique().astype(str).tolist())
    req_exp = sorted(df["required_experience"].dropna().unique().astype(str).tolist())
    req_edu = sorted(df["required_education"].dropna().unique().astype(str).tolist())
    industries = sorted(df["industry"].dropna().unique().astype(str).tolist())
    functions = sorted(df["function"].dropna().unique().astype(str).tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        employment_type = st.selectbox("Employment Type", ["Unknown"] + emp_types)
        required_experience = st.selectbox("Required Experience", ["Unknown"] + req_exp)
        required_education = st.selectbox("Required Education", ["Unknown"] + req_edu)
    with col_b:
        Industry = st.selectbox("Industry", ["Unknown"] + industries)
        Function = st.selectbox("Function", ["Unknown"] + functions)

    telecommuting = st.selectbox("Telecommuting Allowed", [0, 1])
    has_company_logo = st.selectbox("Has Company Logo", [0, 1])
    has_questions = st.selectbox("Has Questions", [0, 1])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "title": title,
            "description": description,
            "requirements": requirements,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": Industry,
            "function": Function,
            "telecommuting": telecommuting,
            "has_company_logo": has_company_logo,
            "has_questions": has_questions
        }])

        input_df["text"] = (
            input_df["title"] + " " +
            input_df["description"] + " " +
            input_df["requirements"]
        )
        def clean_text(t):
            t = str(t).lower()
            t = " ".join(t.split())
            return t
        input_df["text_clean"] = input_df["text"].apply(clean_text)

        X_in_tfidf = tfidf.transform(input_df["text_clean"])
        X_in_cat = ohe.transform(input_df[cat_cols].fillna("Unknown"))
        X_in = np.hstack([X_in_tfidf.toarray(), X_in_cat])

        pred = best_model.predict(X_in)[0]
        proba = best_model.predict_proba(X_in)[0]
        prob_fake = proba[1]

        if pred == 1:
            st.error("🚨 Predicted: FAKE job posting")
            st.write(f"Probability of being fake: **{prob_fake*100:.2f}%**")
            st.warning(
                "This job posting shows patterns similar to known scams. "
                "Be cautious before sharing personal info or paying any fees."
            )
        else:
            st.success("✅ Predicted: REAL job posting")
            st.write(f"Probability of being fake: **{prob_fake*100:.2f}%**")
            st.info(
                "This job looks legitimate based on learned patterns, "
                "but always verify the company and role before applying."
            )