import streamlit as st
import sqlalchemy
import pandas as pd
from sbert import sbert
from sklearn.metrics.pairwise import cosine_similarity

from load_vects import download_or_cached

from pathlib import Path

@st.cache_resource
def connect_db():
    return sqlalchemy.create_engine(st.secrets["DB_URL"])

@st.cache_data
def load_generated(filename):
    local_file = download_or_cached(st.secrets["RELEASES_URL"] + filename)
    return pd.read_feather(local_file)

def weighted_mean(values, weights):
    has_one = False
    value_sum = 0
    weight_sum = 0

    for val, w in zip(values, weights):
        if pd.isna(val):
            continue

        has_one = True
        
        value_sum += val * w
        weight_sum += w
    
    if not has_one:
        return pd.NA

    return value_sum / weight_sum

@st.cache_data(ttl=5*60)
def vectorize(text):
    return sbert.transform(text).reshape(1, -1)

# @st.cache_data
# def fetch_from_db(source, id):
#     engine = connect_db()
#     with engine.begin() as conn:
#         return conn.execute(
#             "SELECT title, abstract, id_doi, id_scopus, source FROM combined_minimal WHERE index = :index",
#             { "id": id, "source": source }
#         )
    
# def fetch_details(df):
#     records = []
#     for i, row in df.iterrows():
#         records.append(fetch_from_db(row.source, row.id))
    
#     return pd.DataFrame.from_records(records, columns=["title", "abstract", "id_doi", "id_scopus", "source"])

@st.cache_data
def fetch_details(df):
    engine = connect_db()
    with engine.begin() as conn:
        return pd.read_sql_query(
            "SELECT * FROM combined_minimal WHERE index = ANY(%s)",
            params=(df.index.to_list(),),
            con=conn
        ).set_index("index").join(df).sort_values("overall_similarity", ascending=False)


TITLE_WEIGHT = 1/4
ABSTRACT_WEIGHT = 3/4

def search(query, title=True, abstract=True):
    query = query.strip()
    if len(query.strip()) == 0:
        return pd.Series([], name="similarity")
    
    return search_vector(vectorize(query), title, abstract)

@st.cache_data(ttl=5*60)
def search_vector(query_vector, title, abstract):
    if title:
        title_sim_np = cosine_similarity(title_vec, query_vector).flatten()
    else:
        title_sim_np = None

    title_similarity = pd.Series(
        title_sim_np,
        index=title_vec.index,
        name="title_similarity"
    )
    
    if abstract:
        abstract_sim_np = cosine_similarity(abstract_vec, query_vector).flatten()
    else:
        abstract_sim_np = None

    abstract_similarity = pd.Series(
        abstract_sim_np,
        index=abstract_vec.index,
        name="abstract_similarity"
    )

    similarity = pd.concat([title_similarity, abstract_similarity], axis=1)

    similarity["overall_similarity"] = similarity.agg(
            lambda x: weighted_mean(
                (x.title_similarity, x.abstract_similarity),
                (TITLE_WEIGHT, ABSTRACT_WEIGHT)
            ),
            axis=1
        )

    return similarity

abstract_vec = load_generated("abstract_vec.feather")
title_vec = load_generated("title_vec.feather")
sim_mtx = load_generated("similarity_mtx.feather")

st.header("🔎 Paper search and suggestions")
with st.form("search", border=True):
    q = st.text_input("Query", placeholder="use of machine learning in linguistics")
    col1, col2 = st.columns([5,1])
    with col1:
        include_title = st.checkbox("Search in titles", value=True)
        include_abstract = st.checkbox("Search in abstracts", value=True)
    with col2:
        st.form_submit_button("Search", type="primary", use_container_width=True, icon=":material/search:")
# st.write()

def show_more():
    st.session_state.show_more_for = q

# st.write(similarities)
# st.write(similarities.idxmax())

# df = df.dropna()
# df.dropna(subset=["id_doi", "bib_abstract"], inplace=True)

SIMILARITY_CRITERION = 0.35
MIN_SIMILARITY = 0.1

if q:
    # similarities = df.join(search(q, include_title, include_abstract)).sort_values("overall_similarity", ascending=False)
    similars = search(q, include_title, include_abstract) \
        .sort_values("overall_similarity", ascending=False) \
        .head(5)
    
    query_vector = vectorize(q) if q else None

    showing_more = st.session_state.get("show_more_for") == q
    if showing_more:
        criteria = similars.overall_similarity > MIN_SIMILARITY
    else:
        criteria = similars.overall_similarity > SIMILARITY_CRITERION
    matches = fetch_details(similars[criteria])
else:
    matches = None

if "paper_recommendations" not in st.session_state:
    st.session_state.paper_recommendations = dict()

if st.session_state.get("q") != q:
    st.session_state.q = q

    # reset prefs
    st.session_state.show_more_for = None
    st.session_state.paper_recommendations.clear()


def toggle_recommendation(paper):
    recommendations = st.session_state.paper_recommendations.get(paper.name)

    if recommendations is None:
        # paper_title_vec = title_vec.loc[paper.name].to_numpy()
        # paper_abstract_vec = abstract_vec.loc[paper.name].to_numpy()
        # # abstract_vec.loc[paper.name].to_numpy() * ABSTRACT_WEIGHT
        # title_sims = search_vector(paper_title_vec.reshape(1, -1), True, False).title_similarity
        # abstract_sims = search_vector(paper_abstract_vec.reshape(1, -1), False, True).abstract_similarity
        # similarities = (title_sims * TITLE_WEIGHT + abstract_sims * ABSTRACT_WEIGHT) \
        #     .sort_values(ascending=False).head(5).rename("overall_similarity")
        # similar_papers = fetch_details(pd.DataFrame(similarities))
        # st.session_state.paper_recommendations[paper.name] = similar_papers

        similarities = sim_mtx.loc[paper.name] \
            .drop(paper.name) \
            .sort_values(ascending=False) \
            .head(5).rename("overall_similarity")
        similar_papers = fetch_details(pd.DataFrame(similarities))
        st.session_state.paper_recommendations[paper.name] = similar_papers
        # recommendations = pd.merge(df, similar, how="inner", left_index=True, right_index=True).sort_values("similarity", ascending=False)
        # st.session_state.paper_recommendations[paper.name] = recommendations.sort_values("similarity", ascending=False).head(5)
    else:
        del st.session_state.paper_recommendations[paper.name]

with st.container():
    if len(q.strip()) == 0:
        st.caption("Search for your paper and find paper suggestions")
    elif matches is not None and len(matches) > 0:
        st.subheader("Results")
        col1, col2, col3 = st.columns([1.5, 9, 2])

        for i, (_, paper) in enumerate(matches.head(5).iterrows()):
            with st.container(border=True):
                col1, col2 = st.columns([11, 1.2], gap="medium")
                with col1:
                    st.text(f"{i+1}. {paper.title}")
                with col2:
                    st.text(f"{round(paper.overall_similarity * 100, 1)}%")

                if paper.abstract is not None:
                    with st.expander("Abstract"):
                        with st.container():
                            st.text(paper.abstract)
                else:
                    st.caption("This paper has no abstract")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"DOI: {paper.id_doi}")
                with col2:
                    if include_title and include_abstract:
                        st.caption(f"Title / abstract match: {paper.title_similarity * 100:.1f}% / {paper.abstract_similarity * 100:.1f}%")
                        
                col1, col2 = st.columns([2.3, 6])
                with col1:
                    already_showing = paper.name in st.session_state.paper_recommendations
                    st.button(
                        "Suggest similar" if not already_showing else "Hide similar",
                        key=i,
                        use_container_width=True,
                        on_click=lambda paper=paper: toggle_recommendation(paper),
                        # disabled=already_showing,
                        type="primary" if not already_showing else "secondary",
                        icon=":material/book_4_spark:" if not already_showing else ":material/visibility_off:"
                    )
                with col2:
                    if paper.source == "scopus":
                        scopus_id = paper.id_scopus.rpartition(":")[-1]
                        st.link_button("Scopus",
                                       url=f"https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp={scopus_id}&origin=inward",
                                       icon=":material/open_in_new:")
                    elif paper.source == "arxiv":
                        st.link_button("Arxiv",
                                       url=f"https://doi.org/{paper.id_doi}",
                                       icon=":material/open_in_new:")
                
                recommendations = st.session_state.paper_recommendations.get(paper.name)
                if recommendations is not None:
                    # st.divider()
                    # st.write(recommendations)
                    st.markdown("#### Similar papers")
                    for i, (_, rec_paper) in enumerate(recommendations.iterrows()):
                        # col1, col2 = st.columns([11, 1.2], gap="medium")
                        # with col1:
                        #     st.text(f"{i+1}. {rec_paper.title}")
                        # with col2:
                        #     st.text(f"")
                        with st.expander(f"{round(rec_paper.overall_similarity * 100, 1)}% — {rec_paper.title}"):
                            st.markdown(f"##### {rec_paper.title}")

                            st.markdown(f"###### Abstract")
                            st.write(rec_paper.abstract)

                            st.caption(f"DOI: {rec_paper.id_doi}")

                            if paper.source == "scopus":
                                scopus_id = paper.id_scopus.rpartition(":")[-1]
                                st.link_button("Scopus",
                                               url=f"https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp={scopus_id}&origin=inward",
                                               icon=":material/open_in_new:")
                            elif paper.source == "arxiv":
                                st.link_button("Arxiv",
                                               url=f"https://doi.org/{paper.id_doi}",
                                               icon=":material/open_in_new:")
        
        if matches.shape[0] < 5:
            st.button("Show less similar matches", help=f"Show papers with similarity < {SIMILARITY_CRITERION*100:.0f}%", on_click=show_more)
    else:
        if showing_more:
            st.text("No match found")
        else:
            st.text("No match found with enough similarity")
        st.button("Show results anyway", on_click=show_more)
# st.write(similarities.max())
# st.text(similarity)
# st.write(matching_paper.title)
# st.text(matching_paper.bib_abstract)

