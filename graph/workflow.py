from graph.state import LegalState
from langgraph.graph import StateGraph, END
from routers.input_router import input_router
from routers.pdf_type_router import pdf_type_router
from routers.query_router import query_router
from agents.legal_pipeline.reader import reader_agent
from agents.legal_pipeline.risk_detector import risk_detector_agent
from agents.legal_pipeline.simplifier import simplifier_agent
from agents.legal_pipeline.summary import summary_agent
from agents.general_chat_agent import general_chat_agent
from agents.legal_expert_agent import legal_expert_agent
from agents.pdf_qa_agent import pdf_qa_agent

def route_by_input_type(state:LegalState):
    return state.get("input_type", "query")

def route_by_pdf_type(state: LegalState):
    return state.get("pdf_type", "normal")

def route_by_query_type(state:LegalState):
    return state.get("query_type", "general")


def build_graph():
    graph = StateGraph(LegalState)
    
    graph.add_node("input_router", input_router)
    graph.add_node("pdf_type_router", pdf_type_router)
    graph.add_node("query_router", query_router)
    graph.add_node("reader_agent", reader_agent)
    graph.add_node("risk_detector_agent", risk_detector_agent)
    graph.add_node("simplifier_agent", simplifier_agent)
    graph.add_node("summary_agent", summary_agent)
    graph.add_node("pdf_qa_agent", pdf_qa_agent)
    graph.add_node("legal_expert_agent", legal_expert_agent)
    graph.add_node("general_chat_agent", general_chat_agent)

    graph.set_entry_point("input_router")
    
    # input_router
    graph.add_conditional_edges(
        "input_router", 
        route_by_input_type,
        {
            "pdf": "pdf_type_router",
            "query": "query_router"
        }
    )
    
    # pdf_type_router 
    graph.add_conditional_edges(
        "pdf_type_router",
        route_by_pdf_type,
        {
            "legal": "reader_agent", 
            "normal": "pdf_qa_agent"
        }
    )
    
    # query_router 
    graph.add_conditional_edges(
        "query_router", 
        route_by_query_type,
        {
            "legal": "legal_expert_agent", 
            "general": "general_chat_agent"
        }
    )
    
    # Legal-pipeline
    graph.add_edge("reader_agent", "risk_detector_agent")
    graph.add_edge("risk_detector_agent", "simplifier_agent")
    graph.add_edge("simplifier_agent", "summary_agent")

    graph.add_edge("summary_agent", END)
    graph.add_edge("pdf_qa_agent", END)
    graph.add_edge("legal_expert_agent", END)
    graph.add_edge("general_chat_agent", END)
    
    return graph.compile()

    
lexai_graph = build_graph()