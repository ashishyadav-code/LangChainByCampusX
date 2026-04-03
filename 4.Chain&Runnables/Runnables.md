# Runnables

Standalone, reusable units of work in LangChain that can be composed together.

## Types

- `RunnableSequence` — chain steps in order
- `RunnableParallel` — run multiple runnables in parallel
- `RunnableBranch` — conditional routing based on input
- `RunnableLambda` — wrap any Python function
- `RunnablePassthrough` — pass input as-is
- `RunnableTryCatch` — handle errors gracefully

## LCEL (LangChain Expression Language)

Build sequential chains using the `|` pipe operator:

```python
chain = prompt | model | output_parser
```