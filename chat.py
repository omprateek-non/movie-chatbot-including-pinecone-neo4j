
import os

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_neo4j import Neo4jGraph

load_dotenv(override=True)

print("\nMOVIE INTELLIGENCE - DATABASE CHECK")
print("=" * 45)


# 1. Initialize Mistral

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

print("\nMistral initialized.")

test_vector = embeddings.embed_query(
    "A thriller movie involving mystery and suspense"
)

print("Embedding dimensions:", len(test_vector))


# 2. Connect Pinecone default namespace

vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

print("\nPinecone connected.")

results = vector_store.similarity_search_with_score(
    "A thriller movie involving mystery and suspense",
    k=3
)

print("\nPINECONE SEARCH RESULTS")

for i, (doc, score) in enumerate(results, 1):

    print(f"\nResult {i}")
    print("Title:", doc.metadata.get("Title"))
    print("Genre:", doc.metadata.get("Genre"))
    print("Score:", score)
    print("Plot:", doc.page_content[:250])


# 3. Connect Neo4j

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j")
)

print("\nNeo4j connected.")


# 4. Check movie count

movie_count = graph.query(
    "MATCH (m:Movie) RETURN count(m) AS total"
)

print("\nNEO4J MOVIE COUNT:", movie_count)


# 5. Test director retrieval

director_results = graph.query(
    """
    MATCH (p:Person)-[:DIRECTED]->(m:Movie)
    WHERE toLower(p.name) CONTAINS toLower($name)
    RETURN p.name AS director,
           m.title AS movie,
           m.releaseYear AS year
    LIMIT 5
    """,
    params={"name": "Nila Madhab Panda"}
)

print("\nDIRECTOR SEARCH RESULTS")

for movie in director_results:
    print(movie)


print("\n" + "=" * 45)
print("DATABASE VALIDATION COMPLETED")
