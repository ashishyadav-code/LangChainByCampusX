from typing import TypedDict, Annotated, Optional, List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

class Review(TypedDict):
    Rating: Annotated[float, "Value between 0 and 5"]
    Pros: Annotated[List[str], "List of advantages"]
    Cons: Annotated[List[str], "List of disadvantages"]
    Summary: str
    Sentiment: Annotated[str, "positive, neutral, or negative"]
    PublisherName: Optional[str]



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,  
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)


structured_model = model.with_structured_output(Review)


text = """
Overall, the iPhone 17e appears to be a well-balanced premium smartphone with strong performance and display capabilities—the A19 chip ensures fast and efficient operation, the 120Hz LTPO display provides a smooth experience, and the camera system is reliable, especially for video; additionally, Apple’s long-term software support is a major advantage. However, there are clear trade-offs: it lacks a telephoto lens, charging speeds are slower compared to many Android competitors, there is no fingerprint sensor, and the price may feel relatively high given the base storage and limited customization. In summary, it is best suited for users who prioritize stability, camera consistency, and ecosystem integration over raw specifications or fast charging.
"""


prompt = f"""
Extract structured data from the following text.

Rules:
- Do NOT hallucinate missing information
- If PublisherName is not mentioned, omit it
- Keep output strictly within schema
- Rating should be inferred reasonably between 0 and 5

Text:
{text}
"""


result = structured_model.invoke(prompt)


print(result)