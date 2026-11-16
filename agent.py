
import os
import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from retrieval import MovieRetriever

load_dotenv(override=True)

retriever = MovieRetriever()

model = init_chat_model(
    "google_genai:gemini-3.5-flash",
    temperature=0,
    max_retries=3
)


@tool
def vector_search(query: str) -> str:
    """Search movies by plot, story, themes and semantic similarity."""

    results = retriever.vector_search(query, k=5)
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def graph_search(entity: str, search_type: str = "director") -> str:
    """Find movies by director, actor, genre or movie name."""

    if search_type not in ["director", "actor", "genre", "movie"]:
        return "Invalid search type."

    results = retriever.graph_search(
        entity=entity,
        search_type=search_type,
        limit=10
    )

    return json.dumps(results, ensure_ascii=False, default=str)


tools = [vector_search, graph_search]
tool_map = {t.name: t for t in tools}

llm = model.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a Bollywood movie research assistant.

        Use vector_search for plot similarity, themes and recommendations.
        Use graph_search for directors, actors, genres and movie relationships.

        Use both tools when necessary.
        Never invent movie details.
        If information is missing, say so.
        Give concise and informative answers.
        The database mainly contains Bollywood movies.
        """
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])


class MovieAgent:

    def ask(self, question, history):

        messages = prompt.invoke({
            "history": history,
            "input": question
        }).to_messages()

        for _ in range(3):

            response = llm.invoke(messages)

            if not response.tool_calls:
                history.extend([
                    HumanMessage(content=question),
                    response
                ])
                return response.content

            messages.append(response)

            for call in response.tool_calls:

                selected_tool = tool_map.get(call["name"])

                try:
                    result = selected_tool.invoke(call["args"])
                except Exception as error:
                    result = f"Tool error: {error}"

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"]
                    )
                )

        final_response = model.invoke(messages)

        history.extend([
            HumanMessage(content=question),
            final_response
        ])

        return final_response.content
