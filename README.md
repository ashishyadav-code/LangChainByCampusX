# LangChain by CampusX

Learning LangChain step by step — from LLMs to chains and structured outputs.

## Structure

| Folder | What it covers |
|--------|---------------|
| `1.LLMs/` | Basic LLM usage with LangChain |
| `2.CHATModels/` | Chat models — Google, HuggingFace, LangChain HF wrapper |
| `3.StructuredOutput&Parsers/` | Structured output using TypedDict, Pydantic, and output parsers |
| `4.Chain&Runnables/` | LCEL, parallel chains, conditional runnables, RunnableBranch |
| `5.EMBEDDINGMODELS/` | Embedding models (in progress) |

## Setup

```bash
python -m venv ai_env
ai_env\Scripts\activate
pip install -r requirements.txt
```

Add your API keys to a `.env` file before running any script.
