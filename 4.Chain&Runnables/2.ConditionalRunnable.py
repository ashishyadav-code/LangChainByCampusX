from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableLambda,RunnableBranch
from typing import Literal
from pydantic import BaseModel,Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)
str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Return either Positive or Negative"
    )

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="CLassigy the sentiment to either positive or negative of following text\n{feedback}\n {formate_instruction}",
    input_variables=["feedback"],
    partial_variables={"formate_instruction": pydantic_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = "Give the appropiate response for this Positive feedback\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template = "Give the appropiate response for this Negative feedback\n{feedback}",
    input_variables=["feedback"]
)

classifier_branch = prompt1 | model | pydantic_parser

Runnable_Branch = RunnableBranch(
    (lambda x:x.sentiment == 'Positive' , prompt2 | model | str_parser),
    (lambda x:x.sentiment == 'Negative' , prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

final_branch = classifier_branch | Runnable_Branch
final_result = final_branch.invoke({"feedback": "this phone is ver laggy and hard to use"})
print(f"Ai: {final_result}")
