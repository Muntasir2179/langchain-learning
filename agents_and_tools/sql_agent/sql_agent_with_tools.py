from dotenv import load_dotenv

# for database
import mysql.connector
from datetime import timedelta, datetime

# for custom tool define
from typing import Optional
from pydantic import BaseModel, Field

# for agent
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.tools import tool


# loading the environment variable
load_dotenv()


# establishing connection to the mysql server
def get_connection():
    conn = mysql.connector.connect(host="localhost",
                                   user="root",
                                   password="",
                                   database="db")
    return conn, conn.cursor()


# schema for insert_data function
class DatabaseInputSchema(BaseModel):
    phone_number: str = Field(description="Should be a valid phone number of 11 digits and it can only starts with 013, 017, 018, 019, 015, 016")
    person_name: str = Field(description="It should be a valid name")
    appointment_date: str = Field(description="It should be a date with the format YY-MM-DD")
    appointment_time: str = Field(description="It will be a time with the format H:M:S")
    age: Optional[int] = Field(description="It will be an integer within the range of 20-100", default=None)


# defining custom tool
@tool("insert_data", args_schema=DatabaseInputSchema, return_direct=True)
def insert_data(phone_number: str, person_name: str, appointment_date: str, appointment_time: str, age: int = None):
    """This function inserts data into database table. Use this function only when you need to insert some data into the database."""
    appointment_time_obj = datetime.strptime(appointment_time, "%H:%M:%S")
    appointment_end_time_obj = appointment_time_obj + timedelta(minutes=5)
    appointment_end_time = appointment_end_time_obj.strftime("%H:%M:%S")

    # SQL query to insert data into the table
    insert_query = f'''
    INSERT INTO mytable (phone_number, person_name, age, appointment_date, appointment_time, appointment_end_time)
    VALUES (%s, %s, %s, %s, %s, %s)
    '''
    
    # Data to insert
    data = (phone_number, person_name, age, appointment_date, appointment_time, appointment_end_time)

    try:
        conn, cursor = get_connection()
        cursor.execute(insert_query, data)
        conn.commit()

        if conn.is_connected():
            cursor.close()
            conn.close()
        return "Data inserted successfully into the table."
    except Exception as e:
        return f"Error occurred during insertion.\nError message:\n{e}"


llm_with_tool = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview").bind_tools(tools=[insert_data])

# building chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an appointment schedule manager. You only response to the question related to appointment management. Your task is to help user book an appointment. "
            "In order to book an appointment for user you will need the following information. Do not execute tool function without complete information."
            "    - Phone number"
            "    - Person name"
            "    - Age"
            "    - Appointment date (reformat it if you need in format YY-MM-DD)"
            "    - Appointment time (reformat it if you need in 12 hour format H:M:S)"
            "Here, age is an optional argument but you should ask user to put age also. If user forgot to provide any information, ask user for those information. You will "
            "not assume any information.\n"
            "In case you face any kind of error message from the tool, you will analyze that error message and generate a feedback response for the user explaining the issue. "
            "Do not show the error message in your feedback response."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"])
    }
    | prompt
    | llm_with_tool
    | ToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=[insert_data], verbose=True)

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

