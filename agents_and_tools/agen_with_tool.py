from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
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
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # placeholder for storing tools response which is called "intermediate_steps"
])

# print(prompt.invoke({"input": "Who is the current head of the Bangladesh government.", "agent_scratchpad": []}))

llm_with_tools = ChatGroq(model="mixtral-8x7b-32768").bind_tools(tools=tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"])
        # we can also use format_to_openai_tool_messages() function to convert tool actions to llm executable messages
    }
    | prompt
    | llm_with_tools
    | ToolsAgentOutputParser()
    # we can also use OpenAIToolsAgentOutputParser() for parsing llm responses
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# streaming the response from the agent
# stream() will show all the function call that the agent will make as we have set verbose=True
for chunk in agent_executor.stream({"input": "How many letters in the word Muntasir?"}):
    print(chunk, end="", flush=True)



'''
Note:
format_to_openai_tool_messages() and format_to_tool_messages() works as same
OpenAIToolsAgentOutputParser() and ToolsAgentOutputParser() works as same

Sequential Processing in the Pipeline:
    1. The lambda function defining "input" pulls the user's original input.
    2. After some steps, the agent interacts with the tools, and the results of those tool calls are added to "intermediate_steps".
    3. When the agent needs to display or process the scratchpad (where intermediate steps are stored), the lambda function for 
       "agent_scratchpad" calls format_to_openai_tool_messages() to prepare those steps for inclusion in the final response.

Conclusion:
Even though the initial input dictionary only has the "input" key, during the agent's execution, it performs intermediate actions 
(like calling tools), and these are stored in "intermediate_steps". The "intermediate_steps" field is populated dynamically as the 
agent processes the request, enabling the lambda function to access and format them when needed.
'''

