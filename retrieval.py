
import os

from annotated_types import doc
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_neo4j import Neo4jGraph

load_dotenv(override=True)


class MovieRetriever:

    def __init__(self):

        # Mistral embeddings

        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        # Pinecone connection

        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        self.index = self.pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        # Neo4j connection

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )

        print("Movie Retriever initialized successfully.")

    # --------------------------------
    # VECTOR SEARCH
    # --------------------------------

    def vector_search(self, query, k=5):

        docs = self.vector_store.similarity_search(
            query,
            k=k *3
        )

        results = []

        seen = set()

        for doc, score in docs:

            title = doc.metadata.get("Title", "Unknown")
            year = doc.metadata.get("Release Year", "Unknown")

            key = (title.strip().lower(), year)

            if key in seen:
                continue

            seen.add(key)

            results.append({
                "title": title,
                "year": year,
                "genre": doc.metadata.get("Genre", "Unknown"),
                "director": doc.metadata.get("Director", "Unknown"),
                "cast": doc.metadata.get("Cast", "Unknown"),
                "plot": doc.metadata.get("Plot", doc.page_content),
                "score": float(score)
            })

            if len(results) >= k:
                break

    # --------------------------------
    # GRAPH SEARCH
    # --------------------------------

    def graph_search(self, entity, search_type="director", limit=10):

        if search_type == "director":

            query = """
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($entity)
            RETURN p.name AS director,
                   m.title AS title,
                   m.releaseYear AS year,
                   m.origin AS origin
            ORDER BY year DESC
            LIMIT $limit
            """

        elif search_type == "actor":

            query = """
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($entity)
            RETURN p.name AS actor,
                   m.title AS title,
                   m.releaseYear AS year,
                   m.origin AS origin
            ORDER BY year DESC
            LIMIT $limit
            """

        elif search_type == "genre":

            query = """
            MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
            WHERE toLower(g.name) CONTAINS toLower($entity)
            RETURN m.title AS title,
                   m.releaseYear AS year,
                   g.name AS genre
            ORDER BY year DESC
            LIMIT $limit
            """

        elif search_type == "movie":

            query = """
            MATCH (m:Movie)
            WHERE toLower(m.title) CONTAINS toLower($entity)
            RETURN m.title AS title,
                   m.releaseYear AS year,
                   m.origin AS origin,
                   m.wikiPage AS wikiPage
            LIMIT $limit
            """

        else:
            raise ValueError("Unsupported search type")

        return self.graph.query(
            query,
            params={
                "entity": entity,
                "limit": limit
            }
        )

    # --------------------------------
    # HYBRID SEARCH
    # --------------------------------

    def hybrid_search(self, query, entity=None, search_type="director"):

        vector_results = self.vector_search(
            query,
            k=10
        )

        graph_results = []

        if entity:

            graph_results = self.graph_search(
                entity,
                search_type=search_type,
                limit=10
            )

        return {
            "vector_results": vector_results,
            "graph_results": graph_results
        }
