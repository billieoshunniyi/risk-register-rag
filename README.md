# Risk Register RAG System

A Retrieval-Augmented Generation (RAG) application for exploring, analysing, and querying organisational risk register data through a Streamlit interface.

## Overview

The Risk Register RAG System combines structured risk register data, semantic retrieval, and a large language model to help users explore documented risks and retrieve relevant information using natural-language questions.

The project demonstrates the integration of data processing, vector search, language models, and interactive visualisation in a practical application.

## Features

- **Risk register exploration:** Work with structured risk data, including risk categories, departments, owners, mitigation actions, and statuses.
- **Natural-language queries:** Ask questions about information contained in the risk register.
- **Retrieval-Augmented Generation:** Retrieve relevant information from the risk register to provide context for generated responses.
- **Vector search:** Use Chroma for vector storage and Hugging Face sentence embeddings for semantic retrieval.
- **Interactive interface:** Use Streamlit to access the application's functionality.
- **Data visualisation:** Use Plotly to support interactive exploration of risk information.

## Technology Stack

| Component | Technology |
|---|---|
| Application interface | Streamlit |
| Data processing | Python, pandas |
| Data visualisation | Plotly |
| Language model integration | LangChain, Groq |
| Vector database | Chroma |
| Text embeddings | Hugging Face Sentence Transformers |
| Configuration | Python dotenv |

## Architecture

1. **Data ingestion:** Load structured risk register records from a CSV file.
2. **Text embeddings:** Convert relevant risk information into vector representations using a sentence-transformer embedding model.
3. **Vector retrieval:** Store and retrieve relevant information through Chroma.
4. **Response generation:** Use retrieved context with a language model to respond to user queries.
5. **User interface:** Present the application's functionality through Streamlit.

## Project Structure

```text
risk-register-rag/
├── data/
│   └── risk_register.csv
├── .gitignore
├── app.py
├── requirements.txt
└── utils.py
```

## Getting Started

### Prerequisites

- Python
- A Groq API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/billieoshunniyi/risk-register-rag.git
   cd risk-register-rag
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API key:

   ```text
   GROQ_API_KEY=your_groq_api_key
   ```

   Replace the placeholder with your own key. Do not commit your `.env` file or expose your API key publicly.

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Limitations and Responsible Use

This application is a decision-support and exploratory analysis tool. Its outputs should be checked against the original risk register and should not replace professional risk assessment or organisational governance processes.

The quality of retrieval and generated responses depends on the source data, retrieval configuration, and language model.

## Future Improvements

Potential areas for further development include:

- Automated evaluation of retrieval relevance and answer quality.
- More comprehensive testing and error handling.
- Deployment with managed secrets and environment configuration.
- Additional risk trends, filters, and analytical visualisations.


