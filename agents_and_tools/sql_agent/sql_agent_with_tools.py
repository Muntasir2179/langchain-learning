from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser

# custom tools and prompt
from custom_tools import insert_data, search_data, update_data, delete_data
from custom_prompts import custom_prompt


# loading the environment variable
load_dotenv()

tools = [
    insert_data,
    search_data,
    update_data,
    delete_data
]

llm_with_tool = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview").bind_tools(tools=tools)

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"])
    }
    | custom_prompt
    | llm_with_tool
    | ToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    response = agent_executor.invoke(input={"input": query, "chat_history": chat_history})

    print(f"\nAgent: {response["output"]}\n")

    # appending user query and agent response to the chat history
    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=response["output"])
        ]
    )

