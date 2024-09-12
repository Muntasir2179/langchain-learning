from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# loading the environment variables
load_dotenv()


PROJECT_ID = "chatbot-ac1cb"
SESSION_ID = "user1_session_new"
COLLECTION_NAME = "chat_history"

# initialize firestore client
print("Initializing Firestore Client....")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat message history
print("Initializing Firestore Chat Message History....")
chat_history = FirestoreChatMessageHistory(session_id=SESSION_ID,
                                           collection=COLLECTION_NAME,
                                           client=client)

print("Chat History Initialized")
print("Current Chat History: ", chat_history.messages)


# loading the chat model
model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

# let's chat with AI
while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
