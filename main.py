from dotenv import load_dotenv

import schema

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import RAW_REACT_PROMPT_INSTRUCTIONS
from schema import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instruction = PromptTemplate(
    template=RAW_REACT_PROMPT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm, tools, prompt=react_prompt_with_format_instruction)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parsed_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parsed_output


def main():
    result = chain.invoke(
        {
            "input": "Search for 3 job postings for an AI engineer using langchain in the Bay Area on Linkedin and list their details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
