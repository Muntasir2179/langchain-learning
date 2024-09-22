from dotenv import load_dotenv

# required imports for tools
from langchain_community.agent_toolkits import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.tools import format_to_tool_messages

# imports for chat models
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# imports for building agents
from langchain.agents import AgentExecutor


# loading the environment variable
load_dotenv()

# loading the gmail credentials
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json"
)
api_resource = build_resource_service(credentials=credentials)


# initializing gmail tools
toolkit = GmailToolkit(api_resource=api_resource)
tools = toolkit.get_tools()

# prompt template for agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert email manager capable of using provided tools to perform tasks that uer might ask you to do."
            "Such tasks can be sending email, preparing draft, searching for specific email."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# loading the LLM and binding the tools
llm_with_tools = ChatGroq(model="llama-3.1-70b-versatile").bind_tools(tools=tools)

# building the agent
agent = (
    {
        "input": lambda x: x["input"],  # fetching the user query
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),  # fetching the response from the tool
        "chat_history": lambda x: x["chat_history"]  # fetching the chat history
    }
    | prompt
    | llm_with_tools
    | ToolsAgentOutputParser()
)

# creating agent executor for executing agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initializing chat history
chat_history = []

# lets chat with our agent
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    response = agent_executor.invoke(input={"input": query, "chat_history": chat_history})
    print(f"\nAgent: {response["output"]}\n")

    # appending query and response to the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))






"""
Navigate to GmailToolkit class and you will find the separate tools for performing gmail operations and their required schema to configure the tools for custom
usecase.
"""