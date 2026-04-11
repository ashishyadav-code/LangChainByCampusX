from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the question: {question} \nContext: {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

loader = WebBaseLoader('https://www.apple.com/in/iphone/compare/')
docs = loader.load()

# Clean + truncate
soup = BeautifulSoup(docs[0].page_content, "html.parser")
text = soup.get_text()[:3000]

chain = prompt | model | parser

print(chain.invoke({
    'question': 'What product is being discussed?',
    'text': text
}))