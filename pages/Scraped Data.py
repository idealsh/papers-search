import streamlit as st
import sqlalchemy
from sqlalchemy.dialects import postgresql
import pandas as pd
import plotly.express as px
import json
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from subject_data import subject_dict, subject_overall_dict, fields_dict
import re
from collections import Counter
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer

st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/Scopus Data.py", label="Scopus Data", icon="📚")
st.page_link("pages/Scraped Data.py", label="Scraped Data", icon="🌐")
st.page_link("pages/Search Papers.py", label="Search (ML)", icon="⚙️")

@st.cache_resource
def connect_db():
    return sqlalchemy.create_engine(st.secrets["DB_URL"])

@st.cache_data
def load_scraped():
    engine = connect_db()

    with engine.begin() as conn:
        return pd.read_sql_table("papers_arxiv", conn)


df_scraped = load_scraped()


st.markdown(
    """
    <style>
    .sticky-note {
        background-color: rgba(190, 190, 210, 0.09); /* Dark blue background */
        border: 1px solid rgba(190, 190, 210, 0.2); /* Slightly darker border for contrast */
        border-radius: 10px;       /* Rounded corners */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Shadow for pop-out effect */
        padding: 1rem;
        width: 100%;
        margin: auto;
        text-align: center;
        # font-family: Arial, sans-serif;
    }
    .sticky-note h1 {
        padding: 0;
        margin: 0.5rem 0 ;
        font-size: 3rem;
    }
    .sticky-note p {
        margin: 0.5rem 0;
        font-size: 1.25rem;
    }
    .sticky-note p1 {
        margin: 5px 10px 0;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_func(options):
    return fields_dict[options]


with st.sidebar:
    st.sidebar.title("Sidebar Menu")
    st.subheader("📚 Select field")
    field = st.selectbox(
        "",
        options=list(fields_dict.keys()),
        format_func=format_func,
        index=0,
        label_visibility="collapsed",
    )
    placeholder = "Select subject..."
    st.write("Field:", format_func(field))

    st.subheader("📅  Year of Publication")
    start, end = st.slider("", 2018, 2023, (2020, 2022), label_visibility="collapsed")
    st.write("From", str(start), "to", str(end))

df_scraped = df_scraped[(df_scraped["year"] >= start) & (df_scraped["year"] <= end)]
if field == "all":
    total_paper = df_scraped.shape[0]
    num_authors = df_scraped["authors"].explode().nunique()
    df_author = df_scraped
else:
    total_paper = df_scraped[df_scraped["field"] == field].shape[0]
    num_authors = (
        df_scraped[df_scraped["field"] == field]["authors"].explode().nunique()
    )
    df_author = df_scraped[df_scraped["field"] == field]

# st.write(field)

st.title("Scraped Data")
st.write("")

st.markdown("")
st.markdown(
    f"""
    <div class="sticky-note">
        <p>📑 Total Scopus Paper</p>
        <h1>{total_paper}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


df2 = df_scraped["field"].value_counts()
st.header("🥇 Percentage of Papers per Field")
tab1, tab2 = st.tabs(["Pie Chart", "Bar Chart"])

with tab1:

    fig = px.pie(
        df2,
        values=df2.values,
        names=df2.index,
        hole=0.4,
        title="Percentage of Papers per Field",
    )
    st.plotly_chart(fig, theme="streamlit")

with tab2:
    df2_percentage = (df2 / df2.sum()) * 100

    fig = px.bar(
        df2_percentage,
        x=df2_percentage.index,
        y=df2_percentage.values,
        labels={"x": "Subject", "y": "Percentage (%)"},
        title="Percentage of Papers per Field",
    )

    fig.update_layout(
        xaxis_title="Subject", yaxis_title="Percentage (%)", template="plotly_white"
    )

    st.plotly_chart(fig, theme="streamlit")

st.markdown("---")

st.markdown(
    f"""
    <div class="sticky-note">
        <h1>{num_authors}</h1>
        <p>🖋️ Number of Authors</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")

st.header("💬 Top 10 Most Frequent Words in Abstracts")

cv = CountVectorizer(stop_words="english")
abstract_vectors = cv.fit_transform(df_author.abstract.dropna())

token_counts = pd.Series(
    data=np.array(abstract_vectors.sum(axis=0))[0], index=cv.get_feature_names_out()
)

top_10_words = token_counts.rename("Frequency").sort_values(ascending=False).head(15)


fig = px.bar(
    top_10_words,
    x="Frequency",
    y=top_10_words.index,
    orientation="h",
    title="for " + format_func(field) + " Field",
    labels={"Frequency": "Word Frequency", "Word": "Word"},
    text="Frequency",
)


fig.update_layout(
    xaxis_title="Frequency",
    yaxis_title="Words",
    yaxis=dict(categoryorder="total ascending"),
    template="plotly_white",
)


st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.header("⌛ Research Papers Trend Over Time")

df_trends = df_scraped.groupby(["year", "field"]).size().reset_index(name="count")
fig = px.line(
    df_trends, x="year", y="count", color="field", title=f"From {start} to {end}"
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("---")


st.header("📎 Top Authors by Number of Papers")
st.write("")
df_exploded = df_scraped.explode("authors")
author_counts = df_exploded.groupby("authors").size().reset_index(name="count")
author_fields = (
    df_exploded.groupby("authors")["field"]
    .apply(lambda x: ", ".join(x.unique()))
    .reset_index()
)
top_authors = (
    author_counts.merge(author_fields, on="authors")
    .sort_values(by="count", ascending=False)
    .head(5)
)

for index, row in top_authors.iterrows():
    st.markdown(
        f"""
    <div style='border: 1px solid #eaeaea; padding: 10px; margin: 5px; border-radius: 5px; background-color: rgba(190, 190, 210, 0.09);'>
        <strong>📖 Author:</strong> {row['authors']} <br>
        <strong>📚 Papers Published:</strong> {row['count']} <br>
        <strong>🎓 Field(s):</strong> {row['field']}
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.write("")

st.markdown("---")
st.header("🔍 Search for Papers by Title")
search_term = st.text_input("Search here")
if search_term:
    search_results = df_scraped[
        df_scraped["title"].str.contains(search_term, case=False, na=False)
    ]
    st.write(f"Found {len(search_results)} results:")
    st.dataframe(search_results[["year", "title", "field"]])

# Footer
st.markdown("---")
st.markdown("### 🎯 Honggege's memebers")
st.write("")
st.write("1. Bhannavit Sripusitto 6638127221")
st.write("2. Sahanont Thammasitboon 6638236021")
st.write("3. Sirikamol Prapaisuwon 6638250721")
st.write("4. Supitcha Juntra 6638253621")
