import pickle
import os
from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain


# with open(f"text_0.txt", encoding="utf-8") as f:
#     long_text_data = f.read()
#
# with open("embedding.pickle", "rb") as file:
#     embedding = pickle.load(file)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
# split_docs = text_splitter.split_text(long_text_data)
# documents = [Document(doc) for doc in split_docs]

module_dir = os.path.dirname(os.path.abspath(__file__))
embedding_file_path = os.path.join(module_dir, 'embedding.pickle')

with open(embedding_file_path, "rb") as file:
    embedding = pickle.load(file)

faiss_file_path = os.path.join(module_dir, 'faiss_index')


class RAGSystem:
    vector_store = FAISS.load_local(
        os.path.join(module_dir, 'faiss_index'),
        embedding,
        allow_dangerous_deserialization=True
    )
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # key = "You_key"
    llm = GigaChat(
        # credentials=key,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        streaming=False,
        verify_ssl_certs=False,
    )
    # llm = OpenAI(
    #     api_key=key,
    #     base_url="http://5.187.4.150/v1/chat/completions",
    #     model="gpt-3.5-turbo",
    # )
    prompt = ChatPromptTemplate.from_template('''
            Ответь на вопрос пользователя. \
            Используй при этом только информацию из контекста. Если в контексте нет \
            информации для ответа, сообщи об этом пользователю.
            Контекст: {context}
            Вопрос: {input}
            Ответ:''')

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
        )

    def get_responce(self, query: str):
        retrieval_chain = create_retrieval_chain(self.embedding_retriever, self.document_chain)
        responce = retrieval_chain.invoke({"input": query})
        return responce["answer"]
