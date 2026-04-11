from langchain_community.document_loaders import PyPDFLoader

loader  = PyPDFLoader('sample_pypdfloader_test.pdf')

docs = loader.load()
# for doc in loader.lazy_load():
#     print(doc.page_content)
print(docs[0].page_content)


"""Lazy Load vs Load

`lazy_load()` is designed for handling large documents efficiently by loading data incrementally (chunk by chunk). This approach reduces memory consumption and allows processing of documents in a streaming-like manner.

In contrast, `load()` reads the entire document into memory at once. It is simpler and works well for smaller documents where memory usage is not a concern.
"""

