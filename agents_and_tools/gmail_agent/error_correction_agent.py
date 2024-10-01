from dotenv import load_dotenv

# required imports for tools
from cutom_tools import CustomSendGmailMessage
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

# custom import
from utils.helper_functions import text_streamer


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
custom_tool = CustomSendGmailMessage(api_resource=api_resource)

# loading the LLM and binding the tools
llm_with_tools = ChatGroq(model="llama-3.1-70b-versatile").bind_tools(tools=[custom_tool])


# prompt template for agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert email manager capable of using provided tools to perform tasks that user might ask you to do."
            "Such tasks can be sending email, preparing draft, searching for specific email. If you encounter any error message from the tool response, generate a "
            "feedback response explaining the error and request for the required information. Do not made up or assume information after encountering any error."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)


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
agent_executor = AgentExecutor(agent=agent, tools=[custom_tool], verbose=True)

# initializing chat history
chat_history = []

# lets chat with our agent
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    response = agent_executor.invoke(input={"input": query, "chat_history": chat_history})

    text_gen = text_streamer(response["output"])
    for item in text_gen:
        print(item, end="", flush=True)
    print()

    # appending query and response to the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))






"""
Key Observations:
    - Agent can use tools multiple times if it told to do so. Sometime, tools can generate error messages due to inaccurate information. To handle this issue, we can
      configure the prompt to change the agent's behavior in a way that it will come up with a constructive feedback message requesting for the required information that
      user may have forgot to mention.

    - We can define our own custom tools for our specific use cases. Custom tool defining enables us to handle exception in a better way.

    - Any error message generated by the custom tool returned as string not as an exception, will be considered as a valid response by llm and it will go on fixing that issue.
      To control this action we have to tell the agent what to do when encounter such situation by the system prompt.
"""