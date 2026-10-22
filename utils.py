import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()



def load_risk_register(filepath: str) -> pd.DataFrame:
    """Load and validate the risk register from a CSV file."""

    df = pd.read_csv(filepath)

    #checking that the CSV contains data
    if df.empty:
        raise ValueError("The risk register is empty.")

    #checking for duplicate column names
    if not df.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names.")

    #checking that required columns exist
    required_columns = [
        "risk_id",
        "category",
        "department",
        "description",
        "likelihood",
        "impact",
        "risk_score",
        "residual_risk_score",
        "owner",
        "mitigation",
        "status",
        "risk_appetite",
        "date_identified",
        "review_date",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    #removing blank rows
    df = df.dropna(how="all")

    if df.empty:
        raise ValueError("The risk register contains no usable rows.")

    #checking for missing values
    if df.isnull().any().any():
        missing_columns = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Missing values found in these columns: {missing_columns}"
        )

    #checking for duplicate risk IDs
    if df["risk_id"].duplicated().any():
        duplicate_ids = (
            df.loc[df["risk_id"].duplicated(keep=False), "risk_id"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Duplicate risk IDs found: {duplicate_ids}")

    
    
    #validating numeric fields
    numeric_columns = [
        "risk_score",
        "residual_risk_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isnull().any():
            raise ValueError(
                f"Column '{column}' contains missing or non-numeric values."
            )

    #validating categorical fields
    categorical_values = {
        "likelihood": {"low", "med", "medium", "high"},
        "impact": {"low", "med", "medium", "high"},
    }

    for column, allowed_values in categorical_values.items():
        df[column] = df[column].astype(str).str.strip()

        invalid_values = (
            set(df[column].str.lower()) - allowed_values
        )

        if invalid_values:
            raise ValueError(
                f"Unexpected values in '{column}': {invalid_values}"
            )

    #validating date fields
    date_columns = ["date_identified", "review_date"]

    for column in date_columns:
        parsed_dates = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        if parsed_dates.isnull().any():
            raise ValueError(
                f"Column '{column}' contains invalid dates."
            )

        df[column] = parsed_dates


    return df



def df_to_documents(df: pd.DataFrame) -> list:
    """Convert each risk register row into a LangChain Document."""
    documents = []
    for _, row in df.iterrows():
        content = f"""
        Risk ID: {row['risk_id']}
        Category: {row['category']}
        Department: {row['department']}
        Description: {row['description']}
        Likelihood: {row['likelihood']}
        Impact: {row['impact']}
        Risk Score: {row['risk_score']}
        Residual Risk Score: {row['residual_risk_score']}
        Owner: {row['owner']}
        Mitigation: {row['mitigation']}
        Status: {row['status']}
        Risk Appetite: {row['risk_appetite']}
        Date Identified: {row['date_identified']}
        Review Date: {row['review_date']}
        """.strip()
        
        doc = Document(
    page_content=content,
    metadata={
        "risk_id": row['risk_id'],
        "category": row['category'],
        "department": row['department'],
        "status": row['status'],
        "risk_score": int(row['risk_score']),
        "residual_risk_score": int(row['residual_risk_score']),
        "risk_appetite": row['risk_appetite']
    }
)
        documents.append(doc)
    
    return documents


def create_vector_store(
    documents: list,
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """Create a Chroma vector store from risk-register documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store


def load_vector_store(
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store


def build_rag_chain(vector_store: Chroma):
    """Build the RAG chain using modern LangChain LCEL syntax."""
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    prompt = PromptTemplate.from_template("""
You are a risk management analyst assistant.
Use the following risk register entries to answer the question clearly and concisely.
Only use information from the provided context.
If the answer is not in the context, say "I could not find that information in the risk register."

Context:
{context}

Question: {question}

Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever