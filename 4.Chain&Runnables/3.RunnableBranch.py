from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"]
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 100,
        RunnableSequence(
            RunnableLambda(lambda x: {"text": x}),
            prompt2,
            model,
            parser
        )
    ),
    RunnableLambda(lambda x: "Report is already short")
)

final_chain = report_gen_chain | branch_chain

final_result = final_chain.invoke({"topic": "the impact of climate change on global agriculture"})
print(f"Final Result: {final_result}")