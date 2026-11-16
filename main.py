
import os
import json

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from retrieval import MovieRetriever

load_dotenv(override=True)

retriever = MovieRetriever()

model = ChatMistralAI(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    temperature=0
)


@tool
def vector_search(query: str) -> str:
    """Find movies by plot, themes, story and semantic similarity."""

    results = retriever.vector_search(query, k=5)
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def graph_search(
    entity: str,
    search_type: str = "director"
) -> str:
    """Find movies by exact director, actor, genre or movie name."""

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

        Use vector_search for plot and theme similarity.
        Use graph_search for directors, actors, genres and movies.

        Use both tools only when necessary.

        Never invent movie information.
        If results are insufficient, say so.
        Give concise, useful answers.
        """
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])


class MovieAgent:

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def ask(self, question):

        messages = prompt.invoke({
            "history": self.history,
            "input": question
        }).to_messages()

        for _ in range(3):

            response = llm.invoke(messages)

            if not response.tool_calls:
                self.history.extend([
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

        # Final response without additional tool calls
        final_response = model.invoke(messages)

        self.history.extend([
            HumanMessage(content=question),
            final_response
        ])

        return final_response.content


if __name__ == "__main__":

    agent = MovieAgent()

    print("\nMovie GraphRAG Assistant")
    print("Type exit to quit | reset to clear memory\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "reset":
            agent.reset()
            print("Memory cleared.")
            continue

        if not question:
            continue

        try:
            answer = agent.ask(question)
            print("\nAssistant:", answer, "\n")

        except Exception as error:
            print("\nError:", error, "\n")
