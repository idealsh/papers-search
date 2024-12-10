import streamlit as st
import sqlalchemy
from sqlalchemy.dialects import postgresql
import pandas as pd
import plotly.express as px
import json
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from subject_data import subject_dict, subject_overall_dict
import re
from collections import Counter
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer


@st.cache_resource
def connect_db():
    return sqlalchemy.create_engine(st.secrets["DB_URL"])


@st.cache_data
def load():
    engine = connect_db()

    with engine.begin() as conn:
        return pd.read_sql_table("papers_scopus", conn)


df = load()


st.html(
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
    """
)


# st.write(df["subject_codes"].map(lambda x :str(x)[1:3]))
df["two_digits"] = df["subject_codes"].map(lambda x: str(x)[1:3])
# st.write(df['subject_names'].map(set).explode().value_counts())
st.title("Scopus Data")


def format_func(options):
    return subject_overall_dict[options]


with st.sidebar:
    st.sidebar.title("Sidebar Menu")
    st.subheader("📚 Select field")
    field = st.selectbox(
        "",
        options=list(subject_overall_dict.keys()),
        format_func=format_func,
        index=0,
        label_visibility="collapsed",
    )
    placeholder = "Select subject..."
    st.write("Field:", format_func(field))

    st.subheader("📅  Year of Publication")
    start, end = st.slider("", 2018, 2023, (2018, 2023), label_visibility="collapsed")
    st.write("From", str(start), "to", str(end))

col1, col2 = st.columns([1, 1])

total_paper = 0
num_authors = 0
num_countries = 0

df = df[(df["bib_pub_year"] >= start) & (df["bib_pub_year"] <= end)]
if field == "0":
    total_paper = df.shape[0]
    num_authors = df["authors"].explode().nunique()
    num_countries = df["affiliation_country"].nunique()
    df_author = df
else:
    total_paper = df[df["two_digits"] == field].shape[0]
    num_authors = df[df["two_digits"] == field]["authors"].explode().nunique()
    num_countries = df[df["two_digits"] == field]["affiliation_country"].nunique()
    df_author = df[df["two_digits"] == field]


st.html(
    f"""
    <div class="sticky-note">
        <p>📑 Total Scopus Paper</p>
        <h1>{total_paper:,}</h1>
    </div>
    """
)

df2 = df["subject_names"].map(set).explode().value_counts()
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

st.divider()

# Number of Paper per Country Choropleth Map
st.header("🌏 Number of Paper per Country Choropleth Map")
countries_df = df["affiliation_country"].dropna().value_counts().reset_index()
countries_df["Log"] = np.log(countries_df["count"])
with open("Custom Geo Data.json", encoding="utf8") as f:
    geojson = json.load(f)
fig = px.choropleth(
    countries_df,
    geojson=geojson,
    color="Log",
    locations="affiliation_country",
    featureidkey="properties.name_long",
    color_continuous_scale="burg",
    hover_data={"count": ":", "Log": False},
    labels={"affiliation_country": "Country", "count": "Count"},
)
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis={
        "colorbar": {
            "title": "Number of Papers",
            "tickvals": [0, 2, 4, 6, 8],
            "ticktext": [1, 7, 54, 403, 2980],
        }
    },
    geo=dict(showframe=False),
)
st.write("Data is displayed in logarithmic scale")
st.plotly_chart(fig)

col1, col2 = st.columns(2)

with col1:
    st.html(
        f"""
    <div class="sticky-note">
        <h1>{num_authors:,}</h1>
        <p>🖋️ Number of Authors</p>
    </div>
    """,
    )

with col2:
    st.html(
        f"""
    <div class="sticky-note">
        <h1>{num_countries:,}</h1>
        <p>📍 Number of Countries</p>
    </div>
    """,
    )

st.divider()

st.header("💬 Top 10 Most Frequent Words in Abstracts")

cv = CountVectorizer(stop_words="english")
abstract_vectors = cv.fit_transform(df_author.bib_abstract.dropna())

token_counts = pd.Series(
    data=np.array(abstract_vectors.sum(axis=0))[0], index=cv.get_feature_names_out()
)

top_10_words = token_counts.rename("Frequency").sort_values(ascending=False).head(15)


fig = px.bar(
    top_10_words,
    x="Frequency",
    y=top_10_words.index,
    orientation="h",
    title="In " + format_func(field),
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

# st.write(word_freq_df)

st.divider()
st.header("🔎 Author Grouped by Paper Count")

df_authors = df_author["authors"].explode()
author_counts = df_authors.value_counts()
group_counts = author_counts.value_counts().sort_index()
paper_count = group_counts.index

fig = px.bar(
    group_counts,
    x=paper_count,
    y=group_counts.values,
    labels={"x": "Paper Count", "y": "Author Count"},
    title="Paper Count per Author",
)

fig.update_layout(
    yaxis=dict(title="Number of Authors", type="log", titlefont=dict(size=12)),
    xaxis_title="Number of Papers",
    title="Number of Authors Grouped by Paper Count (Log Scale) in "
    + format_func(field),
)

st.plotly_chart(fig, theme="streamlit")

st.divider()
st.header("💰 Ranking Fields by Avg. Funding Count")
# ranking fields by avg funding/paper
df2 = df.explode("subject_codes")
df2["subject_codes"] = (df2["subject_codes"].apply(lambda x: x // 100)).astype(str)
df2.drop_duplicates(subset=["title", "subject_codes"], inplace=True)
df2["funding_count"].fillna(value=0, inplace=True)
df2 = df2[["subject_codes", "funding_count"]]
df2 = (
    df2.groupby(by=["subject_codes"])
    .mean()
    .sort_values(by="funding_count", ascending=False)
    .reset_index()
)
df2["Field"] = df2["subject_codes"].map(subject_overall_dict)
# st.write(df2)

c = (
    alt.Chart(df2)
    .mark_bar()
    .encode(
        y=alt.Y("Field", sort=None),
        x=alt.X("funding_count", title="Avg. Funding Count Per Paper"),
    )
)
st.altair_chart(c)
