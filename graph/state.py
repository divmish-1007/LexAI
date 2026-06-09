from typing import TypedDict

class LegalState(TypedDict):
    # Input
    pdf_path: str
    user_query: str
    
    # Routing
    input_type: str
    pdf_type: str
    query_type: str
    
    # PDF processing (legal pipeline)
    raw_text: str
    document_type: str
    clauses: list[str]
    risky_clauses: list[dict]
    simplified_clauses: list[dict]
    final_summary: str
    
    # Conversation
    chat_history: list
    agent_response: str