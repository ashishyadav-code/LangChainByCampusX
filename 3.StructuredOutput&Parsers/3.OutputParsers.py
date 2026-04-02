from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a detailed answer on a cricket player nameed {cricket_player}",
    input_variables=["cricket_player"]
)

template2 = PromptTemplate(
    template="Writet 5 points on given {cricket_player}",
    input_variables=["cricket_player"]
)

parser = StrOutputParser()
prompt = template1.invoke({"cricket_player": "rohit sharma"})

chain = template1 | model | parser | template2 | model | parser

print(f'Analysis: {chain.invoke({"cricket_player": "rohit sharma"})}')

