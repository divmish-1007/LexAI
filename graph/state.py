from langgraph.graph import MessagesState

class AgentState(MessagesState):
    pdf_path:str
    raw_text: str
    document_type: str
    clauses: list[dict]
    risky_clauses: list[dict]
    simplified_clauses: list[dict]
    final_summary: str
    
    user_query: str