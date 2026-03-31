from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

result = model.invoke("what is capital of india, and how many pc games like uncharted are made on indian mythology.")

print(result.text)