import os
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_risk_register,
    df_to_documents,
    create_vector_store,
    load_vector_store,
    build_rag_chain
)

#page configuration
st.set_page_config(
    page_title="Risk Register RAG",
    page_icon="🚧",
    layout="wide"
)

#constants 
DATA_PATH = "data/risk_register.csv"
CHROMA_PATH = "./chroma_db"

#helpers
def risk_colour(score):
    if score >= 7:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"


# Constants
DATA_PATH = "data/risk_register.csv"
CHROMA_BASE_PATH = "./chroma_db"


# Helpers
def risk_colour(score):
    if score >= 7:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"


@st.cache_resource(show_spinner="Loading register and building vector store...")
def initialise(csv_fingerprint):
    df = load_risk_register(DATA_PATH)
    documents = df_to_documents(df)

    #to use a separate database for each version of the CSV
    chroma_path = (
        f"{CHROMA_BASE_PATH}_{csv_fingerprint[:12]}"
    )

    #loading matching index, or build if it doesn't exist
    if os.path.exists(chroma_path):
        vector_store = load_vector_store(chroma_path)
    else:
        vector_store = create_vector_store(
            documents,
            persist_directory=chroma_path
        )

    chain, retriever = build_rag_chain(vector_store)

    return df, chain, retriever


#initialise
with open(DATA_PATH, "rb") as csv_file:
    csv_fingerprint = hashlib.sha256(
        csv_file.read()
    ).hexdigest()

df, chain, retriever = initialise(csv_fingerprint)
#st.write("CSV columns:", df.columns.tolist())

#header
st.title("Risk Register Intellligence")
st.caption("Ask a question * Visualise risk exposure * Filter and Export")

st.divider()

#tabs
tab1, tab2, tab3 = st.tabs(["💡 Chat", "📊 Dashboard", "🗂️ Filter & Export"])

#tab 1 - Chat

with tab1:
    st.subheader("Query your Risk Register")
    st.caption("ex: 'What are our highest priority open risks?' * 'Who owns cybersecurity risks?' * 'What data quality risks exist?'")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    #Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Chat input
    if prompt := st.chat_input("Ask a questions about risks..."):
        #adding user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        #response generation
        with st.chat_message("assistant"):
            with st.spinner("Searching register..."):
                try:
                    response = chain.invoke(prompt)

                    st.markdown(response)

                    # display source documents
                    with st.expander("Source risks used"):
                        docs = retriever.invoke(prompt)

                        if docs:
                            for doc in docs:
                                st.markdown(
                                    f"**{doc.metadata.get('risk_id', 'Unknown ID')}**"
                                    f" - {doc.metadata.get('category', 'Unknown category')}"
                                    f" - Score: {doc.metadata.get('risk_score', 'N/A')}"
                                )
                        else:
                            st.info("No source risks were retrieved.")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception:
                    st.error(
                        "Sorry, I couldn't process your question. "
                        "Please try again in a moment."
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "I couldn't process that question. "
                            "Please try again in a moment."
                        )
                    })
#tab 2 - Dashboard
with tab2:
    st.subheader("Risk Dashboard")

    #KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Risks", len(df))
    with col2:
        high_risks = len(df[df['risk_score'] >= 7])
        st.metric("High Risk 🚩", high_risks)
    with col3:
        open_risks = len(df[df['status'] == 'Open'])
        st.metric("Open Risk", open_risks)  
    with col4:
        mitigated = len(df[df['status'] == 'Mitigated'])
        st.metric("Mitigated ✅️", mitigated)   

    st.divider()

    col_left, col_right = st.columns(2)

    #Risk Heatmap
    with col_left:
        st.subheader("Risk Heatmap")
        likelihood_map = {"Low": 1, "Medium": 2, "High": 3}
        impact_map = {"Low": 1, "Medium": 2, "High": 3} 

        df_heat = df.copy()
        df_heat['likelihood_num'] = df_heat['likelihood'].map(likelihood_map)
        df_heat['impact_num'] = df_heat['impact'].map(impact_map)

        fig_heat = px.scatter(
            df_heat,
            x='impact_num',
            y='likelihood_num',
            size='risk_score',
            color='category',
            hover_data=['risk_id', 'description', 'owner', 'status'],
            labels={
                'impact_num': 'Impact',
                'likelihood_num': 'Likelihood'
            },
            title="Risk Exposure Map"
        )
        fig_heat.update_layout(
            xaxis=dict(tickvals=[1,2,3], ticktext=["Low", "Medium", "High"]),
            yaxis=dict(tickvals=[1,2,3], ticktext=["Low", "Medium", "High"])
        )
        st.plotly_chart(fig_heat, use_container_width=True)
                        
    #Risks by Category
    with col_right:
        st.subheader("Risks by Category")
        category_counts = df.groupby('category').agg(
            count=('risk_id', 'count'),
            avg_score=('risk_score', 'mean')
        ).reset_index()

        fig_cat = px.bar(
            category_counts,
            x='category',
            y='count',
            color='avg_score',
            color_continuous_scale='RdYlGn_r',
            title="Risk Count by Category",
            labels={'count': 'Number of Risks', 'category': 'Category', 'avg_score': 'Average Score'}
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    col_left2, col_right2 = st.columns(2)     

    #Status Donut
    with col_left2:
        st.subheader("Risk Status")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']

        fig_donut = px.pie(
            status_counts,
            values='count',
            names='status',
            hole=0.4,
            title="Risk Status Distribution",
            color_discrete_map={
                'Open': '#EF4444',
                'In Progress': '#F59E0B',
                'Mitigated' : '#10B981'
            }
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    #Top 10 Risks Table
    with col_right2:
        st.subheader("Top 10 Risks by Score")
        top_risks = df.nlargest(10, 'risk_score')[
            ['risk_id', 'category', 'description', 'risk_score', 'status', 'owner']
        ].copy()
        top_risks['risk_level'] = top_risks['risk_score'].apply(risk_colour)
        st.dataframe(
            top_risks[['risk_id', 'category', 'risk_score', 'risk_level', 'status', 'owner']],
            use_container_width=True,
            hide_index=True
        )

        st.divider()
    col_left3, col_right3 = st.columns(2)

    # ── Residual vs Inherent Risk ──
    with col_left3:
        st.subheader("Inherent vs Residual Risk")
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            name='Inherent Risk Score',
            x=df['risk_id'],
            y=df['risk_score'],
            marker_color='#EF4444'
        ))
        fig_res.add_trace(go.Bar(
            name='Residual Risk Score',
            x=df['risk_id'],
            y=df['residual_risk_score'],
            marker_color='#10B981'
        ))
        fig_res.update_layout(
            barmode='group',
            xaxis_title="Risk ID",
            yaxis_title="Score",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Risk Appetite Distribution ──
    with col_right3:
        st.subheader("Risk Appetite Distribution")
        appetite_counts = df['risk_appetite'].value_counts().reset_index()
        appetite_counts.columns = ['appetite', 'count']
        colour_map = {
            'Tolerate': '#10B981',
            'Treat': '#F59E0B',
            'Transfer': '#3B82F6',
            'Terminate': '#EF4444'
        }
        fig_app = px.bar(
            appetite_counts,
            x='appetite',
            y='count',
            color='appetite',
            color_discrete_map=colour_map,
            title="Risk Appetite by Category",
            labels={'appetite': 'Risk Appetite', 'count': 'Number of Risks'}
        )
        st.plotly_chart(fig_app, use_container_width=True)

    st.divider()
    
    # ── Risks Due for Review ──
    st.subheader("⚠️ Risks Due for Review")
    import datetime
    today = datetime.date.today()
    df['review_date_parsed'] = pd.to_datetime(df['review_date']).dt.date
    overdue = df[df['review_date_parsed'] <= today][
        ['risk_id', 'category', 'description', 'risk_score', 'owner', 'review_date', 'status']
    ].copy()
    overdue['risk_level'] = overdue['risk_score'].apply(risk_colour)
    if len(overdue) > 0:
        st.warning(f"{len(overdue)} risks are due or overdue for review")
        st.dataframe(overdue, use_container_width=True, hide_index=True)
    else:
        st.success("No risks currently due for review")    

#tab 3 - Filter & Export
with tab3:
    st.subheader("Filter & Export")
    
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    
    with col_f1:
        categories = ["All"] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Category", categories)
    
    with col_f2:
        statuses = ["All"] + sorted(df['status'].unique().tolist())
        selected_status = st.selectbox("Status", statuses)
    
    with col_f3:
        min_score = st.slider("Minimum Risk Score", 1, 9, 1)
    
    with col_f4:
        owners = ["All"] + sorted(df['owner'].unique().tolist())
        selected_owner = st.selectbox("Owner", owners)

    with col_f5:
        departments = ["All"] + sorted(df['department'].unique().tolist())
        selected_dept = st.selectbox("Department", departments)    
    


    # Apply filters
    filtered = df.copy()
    if selected_category != "All":
        filtered = filtered[filtered['category'] == selected_category]
    if selected_status != "All":
        filtered = filtered[filtered['status'] == selected_status]
    if selected_owner != "All":
        filtered = filtered[filtered['owner'] == selected_owner]
    filtered = filtered[filtered['risk_score'] >= min_score]
    if selected_dept != "All":
            filtered = filtered[filtered['department'] == selected_dept]

    st.caption(f"Showing {len(filtered)} of {len(df)} risks")
    
    filtered['risk_level'] = filtered['risk_score'].apply(risk_colour)
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Export
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download filtered risks as CSV",
        data=csv,
        file_name="filtered_risks.csv",
        mime="text/csv"
    ) 

                   