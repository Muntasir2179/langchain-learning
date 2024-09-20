from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv()


# defining tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)

# listing available tools
tools = [get_word_length]

# defining prompt template
store = {}  # storing message history as session key
prompt = ChatPromptTemplate.from_messages([
    (
        # system message
        "system",
        "You are very powerful assistant, but you do not know about current events."
    ),
    MessagesPlaceholder(variable_name='history'),  # placeholder for inserting chat history
    ("user", "{input}"),  # user query
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # placeholder for storing tools response which is called "intermediate_steps"
])


# initializing llm with tools
llm_with_tools = ChatGroq(model="mixtral-8x7b-32768").bind_tools(tools=tools)

# building agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
        # we can also use format_to_openai_tool_messages() function to convert tool actions to llm executable messages
        "history": lambda x: x.get("history", [])
    }
    | prompt
    | llm_with_tools
    | ToolsAgentOutputParser()
    # we can also use OpenAIToolsAgentOutputParser() for parsing llm responses
)

# creating agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# function to fetch chat history if exist otherwise initialize as new chat session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# creating history aware agent
with_message_history = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# let's chat with the agent
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = with_message_history.invoke(
        input={"input": query, "history": []},
        config={"configurable": {"session_id": "some_session_id"}}
        # changing the session id value will make agent start fresh new chat with no previous conversations
    )

    print(f"\nAgent: {response["output"]}\n")






'''
Note:
    * format_to_openai_tool_messages() and format_to_tool_messages() works as same
    * OpenAIToolsAgentOutputParser() and ToolsAgentOutputParser() works as same
    * `store` dictionary contains chat history of different sessions. When the session changes the model will not able to 
      remember previous questions and answers. In that case it will be a fresh chat with no previous conversations and the new session 
      id will be saved as a key in `store` dictionary and the conversations of this current chat will be stored as a value of that key.

Sequential Processing in the agent creation pipeline:
    1. The lambda function defining "input" pulls the user's original input.
    2. After some steps, the agent interacts with the tools, and the results of those tool calls are added to "intermediate_steps".
    3. When the agent needs to display or process the scratchpad (where intermediate steps are stored), the lambda function for 
       "agent_scratchpad" calls format_to_openai_tool_messages() to prepare those steps for inclusion in the final response.

Conclusion:
Even though the initial input dictionary only has the "input" key, during the agent's execution, it performs intermediate actions 
(like calling tools), and these are stored in "intermediate_steps". The "intermediate_steps" field is populated dynamically as the 
agent processes the request, enabling the lambda function to access and format them when needed.
'''

