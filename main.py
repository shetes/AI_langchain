from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schema import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Use the latest `create_agent` interface with structured output.
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 job postings for an AI engineer using langchain in the Bay Area on Linkedin and list their details.",
                }
            ]
        }
    )

    # When using `response_format`, the result will usually expose a structured response.
    structured = result.get("structured_response", None) # getattr(result, "structured_response", None)
    print(structured)


if __name__ == "__main__":
    main()
