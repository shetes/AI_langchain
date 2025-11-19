from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
chain = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    result = chain.invoke(
        {
            "input": "Search for 3 job postings for an AI engineer using langchain in the Bay Area on Linkedin and list their details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
