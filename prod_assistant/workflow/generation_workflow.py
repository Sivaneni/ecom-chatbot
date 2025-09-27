from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retreiver.retrieval_bckp import Retriever
from utils.model_loader import ModelLoader
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

retriever_obj = Retriever()
model_loader = ModelLoader()


def format_docs(docs) -> str:
    """Format retrieved documents into a structured text block for the prompt."""
    if not docs:
        return "No relevant documents found."

    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)

    return "\n\n---\n\n".join(formatted_chunks)

def _format_docs(docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        
        meta = docs.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{docs.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)


def build_chain(query):
    """Build the RAG pipeline chain with retriever, prompt, LLM, and parser."""
    retriever = retriever_obj.load_retriever()
    retrieved_docs=retriever.invoke(query)
    
    #retrieved_contexts = [format_docs(doc) for doc in retrieved_docs]
    retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]
    
    llm = model_loader.load_llm()
    prompt = ChatPromptTemplate.from_template(
        PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain,retrieved_contexts


def invoke_chain(query: str, debug: bool = False):
    """Run the chain with a user query."""
    chain,retrieved_contexts = build_chain(query)

    if debug:
        # For debugging: show docs retrieved before passing to LLM
        docs = retriever_obj.load_retriever().invoke(query)
        print("\nRetrieved Documents:")
        print(format_docs(docs))
        print("\n---\n")

    response = chain.invoke(query)
    
    return retrieved_contexts,response

def invoke_chain_v3(query: dict, debug: bool = False):
    """Run the chain with a user query."""
    # Step 4: Prompt and LLM
    prompt = PromptTemplate.from_template("""
    Answer the question based on the context provided.

    Context:
    {context}

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm=ModelLoader().load_llm(), prompt=prompt)
    rag_chain = create_retrieval_chain(retriever_obj.load_retriever(), document_chain)

    if debug:
        # For debugging: show docs retrieved before passing to LLM
        docs = retriever_obj.load_retriever().invoke(query)
        print("\nRetrieved Documents:")
        print(format_docs(docs))
        print("\n---\n")

    response = rag_chain.invoke(query)
    print("response from rag v3 chain",response)
    retrieved_contexts = [_format_docs(doc) for doc in response['context']]
    
    return retrieved_contexts,response


if __name__=='__main__':
    user_query = "what are the postive reveiws provided for dell laptop?"

    retrieved_contexts,response = invoke_chain(user_query)
    
    #this is not an actual output this have been written to test the pipeline
    #response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."
    
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)
    
    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)
     
    #retriever_obj = Retriever()
    
    #retrieved_docs = retriever_obj.call_retriever(user_query)
    
    # def _format_docs(docs) -> str:
    #     if not docs:
    #         return "No relevant documents found."
    #     formatted_chunks = []
    #     for d in docs:
    #         print(d)
    #         meta = d.metadata or {}
    #         formatted = (
    #             f"Title: {meta.get('product_title', 'N/A')}\n"
    #             f"Price: {meta.get('price', 'N/A')}\n"
    #             f"Rating: {meta.get('rating', 'N/A')}\n"
    #             f"Reviews:\n{d.page_content.strip()}"
    #         )
    #         formatted_chunks.append(formatted)
    #     return "\n\n---\n\n".join(formatted_chunks)
    
    # retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]
    # question={"input": user_query}
    # retrieved_contexts,response = invoke_chain_v3(question)
    # print("response coming from llm",response)
    # print("\n--- Retrieved Contexts start---")
    # print(retrieved_contexts)
    # print("\n--- Retrieved Contexts end---")
    
    # #this is not an actual output this have been written to test the pipeline
    # #response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."
    # input=question["input"]
    # print("input question",type(input))
    # context_score = evaluate_context_precision(input,response['answer'],retrieved_contexts)
    # relevancy_score = evaluate_response_relevancy(input,response['answer'],retrieved_contexts)
    
    # print("\n--- Evaluation Metrics ---")
    # print("Context Precision Score:", context_score)
    # print("Response Relevancy Score:", relevancy_score)
    
    
    
# if __name__ == "__main__":
#     try:
#         answer = invoke_chain("can you tell me the price of the iPhone 15?")
#         print("\n Assistant Answer:\n", answer)
#     except Exception as e:
#         import traceback
#         print("Exception occurred:", str(e))
#         traceback.print_exc()