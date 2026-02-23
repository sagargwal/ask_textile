from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    # other params...
)

messages = "tell me about lucknow" 
ai_msg = llm.invoke(messages)
print(ai_msg.content)