import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor


load_dotenv()


# defining tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)

# listing available tools
tools = [get_word_length]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are very powerful assistant, but you do not know about current events."
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# print(prompt.invoke({"input": "Who is the current head of the Bangladesh government.", "agent_scratchpad": []}))

llm_with_tools = ChatGroq(model="mixtral-8x7b-32768").bind_tools(tools=tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# streaming the response from the agent
# stream() will show all the function call that the agent will make as we have set verbose=True
for chunk in agent_executor.stream({"input": "How many letters in the word Muntasir?"}):
    print(chunk, end="", flush=True)
