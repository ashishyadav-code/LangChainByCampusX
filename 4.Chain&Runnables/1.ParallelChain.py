from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm)

llm2 = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    temperature=0.2,
)

model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Genarete a short and simple notes from given topic: {text}",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="Genarete 5 questions and answers from following text: {text}",
    input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="merge the provided notes and quiz into a single docs\n notes ->{notes} , quiz-> {quiz}",
    input_variables=["notes","quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes":prompt1 | model | parser,
    "quiz":prompt2 | model2 | parser
})

merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain

text = """
Artificial Intelligence (AI) is transforming the way humans interact with technology and solve complex problems across multiple domains. It refers to the simulation of human intelligence in machines that are programmed to think, learn, and make decisions. Modern AI systems rely heavily on data, algorithms, and computational power to perform tasks such as natural language processing, image recognition, and predictive analytics. One of the key branches of AI is machine learning, where models improve their performance over time by learning from data without being explicitly programmed. Another emerging area is generative AI, which can create new content such as text, images, music, and even code. Despite its advantages, AI also raises important ethical concerns, including bias, privacy, and job displacement. As AI continues to evolve, it is crucial for developers, policymakers, and society to work together to ensure responsible and fair usage of these powerful technologies.
"""

result = chain.invoke({"text": text})

print(f"Result: {result}")
# chain.get_graph().print_ascii()