# import langchain

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task  = "text-generation"
)

chat_model  = ChatHuggingFace(llm =llm)

result = chat_model.invoke("tell me about lucknow")
print(result.content)