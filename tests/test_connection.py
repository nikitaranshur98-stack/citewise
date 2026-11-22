from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

response = llm.invoke("Say hello in one short sentence.")
print(response.text)