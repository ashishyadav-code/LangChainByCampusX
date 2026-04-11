from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def create_documents():
    return [
        Document(
            page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
            metadata={"team": "Royal Challengers Bangalore"}
        ),
        Document(
            page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
            metadata={"team": "Mumbai Indians"}
        ),
        Document(
            page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
            metadata={"team": "Chennai Super Kings"}
        ),
        Document(
            page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
            metadata={"team": "Mumbai Indians"}
        ),
        Document(
            page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
            metadata={"team": "Chennai Super Kings"}
        )
    ]


def load_embedding():
    return OllamaEmbeddings(
        model="embeddinggemma:latest",
        base_url="http://localhost:11434"
    )


def create_vector_db(documents, embedding):
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="my_chroma_db",
        collection_name="ipl_players"
    )
    db.persist()

def search(db, query, k=3):
    results = db.similarity_search(query, k=k)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print(doc.page_content)
        print("Team:", doc.metadata["team"])
        print("-" * 50)


if __name__ == "__main__":
    docs = create_documents()
    embedding = load_embedding()
    db = create_vector_db(docs, embedding)

    query = "Who is the best IPL captain?"
    search(db, query)