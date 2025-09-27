import os
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retreiver.retrieval import Retriever
from utils.model_loader import ModelLoader
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.api_key_mgr = self.model_loader.api_key_mgr
        self.llm = self.model_loader.load_llm()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        self._save_workflow()

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
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

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        print("[_ai_assistant] State :", state)
        messages = state["messages"]

        last_message = messages[-1].content
        

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        print("[_vector_retriever] State :", state)
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        print("[_vector_retriever] Context :", context)
        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        print("[_grade_documents] State :", state)
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        print("[_grade_documents] Score :", score)
        if "yes" in score.lower():
            return "generator" 
        elif "no" in score.lower():
            return "tavilysearch" 
        else:
            return "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        print("[_generate] State :", state)
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        print("[_rewrite] State :", state)
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}
    
    def _tavilysearch(self, state: AgentState):
        print("--- TAVILY SEARCH ---")
        print("[_tavilysearch] State :", state)
        question = state["messages"][0].content
        tavily_search_tool = TavilySearch(
                max_results=5,
                topic="general",
                    api_key=self.api_key_mgr.get("TAVILY_API_KEY"))
        agent = create_react_agent( self.llm, [tavily_search_tool])
        for step in agent.stream({"messages": question},
            stream_mode="values"):
            
                step["messages"][-1].pretty_print()
        final_answer = step["messages"][-1].content
        return {"messages": [HumanMessage(content=final_answer)]}
        

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        """Build the workflow graph."""
        
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("TavilySearch", self._tavilysearch)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else "tavilysearch",
            {"Retriever": "Retriever","tavilysearch": "TavilySearch"}
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter", "tavilysearch": "TavilySearch"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        # workflow.add_conditional_edges(
        #     "Assistant",
        #     lambda state: "TavilySearch" if "No relevant documents found." in state["messages"][-1].content else END,
        #     {"TavilySearch": "TavilySearch", END: END},
        # )
        workflow.add_edge("TavilySearch", END)
        return workflow
    
    def _save_workflow(self, filename: str = "workflow.png"):
        """Save the workflow graph to a text file."""
        #if file present dont run again
        if os.path.exists(filename):
            print("Workflow already exists")
            return "file already exists"
        else:
            png_bytes = self._build_workflow().compile().get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(png_bytes)

        
        

    # ---------- Public Run ----------
    def run(self, query: str) -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content


if __name__ == "__main__":
    rag_agent = AgenticRAG()
    #answer = rag_agent.run("What is the issue between india vs pakistan in asia cup?")
    #answer = rag_agent.run("What is the price of iPhone 15?")
    answer= rag_agent.run("Can you tell me the price and reviews of oneplus 13?")
    print("\nFinal Answer:\n", answer) 