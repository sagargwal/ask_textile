# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma


# def get_retriver():

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     vectorstore = Chroma(
#         embedding_function=embeddings,
#         persist_directory="./textile_nomic_db"
#     )
#     main_retriever = vectorstore.as_retriever(search_type= "similarity", search_kwargs = {"k":2})

#     return main_retriever

# new = get_retriver()
# query = "what is warf"
# vec_retriver = new.similarity_search(query,k=4)

# # query = "what is yarn strength"

# result = vec_retriver.invoke(query)

# for i, result in enumerate(vec_retriver):
#   print(f"------ result{i}------- \n")
#   print(f"{result.page_content}")

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def get_retriver():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./textile_nomic_db",
        collection_name="textile_engineering"  # ← add this
    )
    main_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return main_retriever
# # Get retriever
# retriever = get_retriever()

# # Query
# query = "what is warp"

# # ── Option 1 — using retriever.invoke() ───────────────────
# results = retriever.invoke(query)

# for i, doc in enumerate(results):
#     print(f"\n------ Result {i+1} -------")
#     print(f"Course:  {doc.metadata['course_title']}")
#     print(f"Lecture: {doc.metadata['lecture_name']}")
#     print(f"Source:  {doc.metadata['source_type']}")
#     print(f"Text:    {doc.page_content[:300]}")