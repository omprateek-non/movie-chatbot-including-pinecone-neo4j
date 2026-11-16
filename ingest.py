
import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import MistralAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "graphdb")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

CSV_PATH = "wiki_movie_plots_deduped.csv"
LIMIT = 1000

for name, value in {
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
}.items():
    if not value:
        raise ValueError(f"{name} is missing from .env")

print("Initializing connections...")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY,
)

stats = index.describe_index_stats()

if stats.dimension != 1024:
    raise ValueError(
        f"Expected 1024 dimensions, found {stats.dimension}"
    )

print("Neo4j connected")
print("Pinecone connected")
print("Pinecone vectors:", stats.total_vector_count)

graph.query(
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (m:Movie) REQUIRE m.title IS UNIQUE"
)

graph.query(
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (p:Person) REQUIRE p.name IS UNIQUE"
)

graph.query(
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (g:Genre) REQUIRE g.name IS UNIQUE"
)

print("Neo4j schema initialized")

loader = CSVLoader(
    file_path=CSV_PATH,
    encoding="utf-8",
    content_columns=["Plot"],
    metadata_columns=[
        "Release Year",
        "Title",
        "Origin/Ethnicity",
        "Director",
        "Cast",
        "Genre",
        "Wiki Page",
    ],
)

docs = loader.load()

print("Total CSV records:", len(docs))

hindi_docs = [
    doc for doc in docs
    if doc.metadata.get("Origin/Ethnicity", "").strip().lower()
    == "bollywood"
    and doc.metadata.get("Title", "").strip()
    and doc.page_content.strip()
]

hindi_docs = hindi_docs[:LIMIT]

print("Bollywood records selected:", len(hindi_docs))

if not hindi_docs:
    raise ValueError("No Bollywood records found")

def parse_names(value):
    if not value or value.strip().lower() == "unknown":
        return []

    return [
        name.strip()
        for name in value.split(",")
        if name.strip()
        and name.strip().lower() != "unknown"
    ]

GRAPH_QUERY = """
UNWIND $rows AS row

MERGE (m:Movie {title: row.title})
SET m.releaseYear = row.year,
    m.origin = row.origin,
    m.wikiPage = row.wikiPage

MERGE (g:Genre {name: row.genre})
MERGE (m)-[:HAS_GENRE]->(g)

WITH row, m

FOREACH (dName IN row.directors |
    MERGE (d:Person {name: dName})
    MERGE (d)-[:DIRECTED]->(m)
)

WITH row, m

FOREACH (aName IN row.actors |
    MERGE (a:Person {name: aName})
    MERGE (a)-[:ACTED_IN]->(m)
)
"""

print("\nStarting Neo4j ingestion...")

graph_batch_size = 100

for start in range(0, len(hindi_docs), graph_batch_size):

    batch = hindi_docs[start:start + graph_batch_size]
    rows = []

    for doc in batch:
        meta = doc.metadata

        year_text = str(
            meta.get("Release Year", "")
        ).strip()

        rows.append({
            "title": meta.get("Title", "").strip(),
            "year": int(year_text) if year_text.isdigit() else None,
            "origin": meta.get("Origin/Ethnicity", "Unknown"),
            "wikiPage": meta.get("Wiki Page", ""),
            "genre": (
                meta.get("Genre", "").strip().lower()
                or "unknown"
            ),
            "directors": parse_names(
                meta.get("Director", "")
            ),
            "actors": parse_names(
                meta.get("Cast", "")
            ),
        })

    graph.query(GRAPH_QUERY, params={"rows": rows})

    print(
        f"Neo4j: {min(start + graph_batch_size, len(hindi_docs))}"
        f"/{len(hindi_docs)} movies processed"
    )

neo4j_result = graph.query(
    "MATCH (m:Movie) RETURN count(m) AS total"
)

print("Neo4j movie count:", neo4j_result[0]["total"])

print("\nStarting plot splitting...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

plot_chunks = splitter.split_documents(hindi_docs)

print("Total plot chunks:", len(plot_chunks))

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
)

print("\nStarting Pinecone ingestion...")

vector_batch_size = 16

for start in range(0, len(plot_chunks), vector_batch_size):

    batch = plot_chunks[start:start + vector_batch_size]

    ids = []

    for doc in batch:
        meta = doc.metadata

        identity = (
            str(meta.get("row", ""))
            + "|"
            + meta.get("Title", "")
            + "|"
            + doc.page_content
        )

        vector_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        ids.append(vector_id)

    print(
        f"Generating embeddings for "
        f"{start + 1}-{min(start + vector_batch_size, len(plot_chunks))}"
        f"/{len(plot_chunks)}"
    )

    vector_store.add_documents(
        documents=batch,
        ids=ids,
    )

    print(
        f"Pinecone uploaded: "
        f"{min(start + vector_batch_size, len(plot_chunks))}"
        f"/{len(plot_chunks)}"
    )

print("\nVerifying databases...")

final_stats = index.describe_index_stats()

print("Neo4j movies:", graph.query(
    "MATCH (m:Movie) RETURN count(m) AS total"
)[0]["total"])

print("Pinecone vectors:", final_stats.total_vector_count)

print("\nTesting vector search...")

results = vector_store.similarity_search(
    "A movie about drought and farmers",
    k=3,
)

for doc in results:
    print("\nMovie:", doc.metadata.get("Title"))
    print("Plot:", doc.page_content[:200])

print("\nINGESTION COMPLETED SUCCESSFULLY")

