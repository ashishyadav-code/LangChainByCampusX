from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Suggest a fictional charachter name based on boruto manga and its clothing and pose for createing image \n,{formate_instruction}",
    input_variables=[],
    partial_variables={'formate_instruction': parser.get_format_instructions()}
)

promt = template.format()
result = model.invoke(promt)
final = parser.parse(result.content)

print(final)