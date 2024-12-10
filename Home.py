import streamlit as st
import sqlalchemy
import pandas as pd
import plotly.express as px
from subject_data import subject_overall_dict


@st.cache_resource
def connect_db():
    return sqlalchemy.create_engine(st.secrets["DB_URL"])


@st.cache_data
def load():
    engine = connect_db()

    with engine.begin() as conn:
        return pd.read_sql_table("papers_scopus", conn)


df = load()


@st.cache_data
def load_scraped():
    engine = connect_db()

    with engine.begin() as conn:
        return pd.read_sql_table("papers_arxiv", conn)


df_scraped = load_scraped()


# Sticky note styling
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


# st.write(df["subject_codes"].map(lambda x :str(x)[1:3]))
df["two_digits"] = df["subject_codes"].map(lambda x: str(x)[1:3])
# st.write(df['subject_names'].map(set).explode().value_counts())
st.title("Datasci Project")
st.write("")
# st.header("Organize your app with layouts")


# Sidebar for Navigation or Settings
def format_func(options):
    return subject_overall_dict[options]


with st.sidebar:
    st.sidebar.title("Sidebar Menu")

    st.subheader("📅  Year of Publication")
    start, end = st.slider("", 2018, 2023, (2020, 2022), label_visibility="collapsed")
    st.write("From", str(start), "to", str(end))


# Main Content Layout:material/fdssdf:
st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/Scopus Data.py", label="Scopus Data", icon="📚")
st.page_link("pages/Scraped Data.py", label="Scraped Data", icon="🌐")
st.page_link("pages/Search Papers.py", label="Search (ML)", icon="⚙️")

col1, col2 = st.columns([1, 1])

total_paper = 0
num_authors = 0
num_countries = 0

df = df[(df["bib_pub_year"] >= start) & (df["bib_pub_year"] <= end)]
df_scraped = df_scraped[(df_scraped["year"] >= start) & (df_scraped["year"] <= end)]

total_paper = df.shape[0]
num_authors = df["authors"].explode().nunique()
num_countries = df["affiliation_country"].nunique()
df_author = df

total_scraped_paper = df_scraped.shape[0]

# Number of Papers

with col1:
    st.markdown(
        f"""
    <div class="sticky-note">
        <p>📑 Total Scopus Paper</p>
        <h1>{total_paper}</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="sticky-note">
        <p>📝 Total Scraped Paper</p>
        <h1>{total_scraped_paper}</h1>
        
    </div>
    """,
        unsafe_allow_html=True,
    )
st.write("")

combind_df = pd.DataFrame(
    {
        "Name": ["Scopus Paper", "Scraped Paper"],
        "Paper": [total_paper, total_scraped_paper],
    }
)

# pie
with st.container():
    fig = px.pie(combind_df, values="Paper", names="Name", title="Number of Paper")
    st.plotly_chart(fig, theme=None)

st.markdown("---")
st.markdown("### 🎯 Honggege's memebers")
st.write("")
st.write("1. Bhannavit Sripusitto 6638127221")
st.write("2. Sahanont Thammasitboon 6638236021")
st.write("3. Sirikamol Prapaisuwon 6638250721")
st.write("4. Supitcha Juntra 6638253621")
