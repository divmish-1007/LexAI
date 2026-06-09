from graph.state import LegalState
from config.llm import llm

def query_router(state: LegalState):
    user_query = state.get("user_query", "")
    
    prompt = f""" You are a query classifier for a Legal AI Assistant.
    
    Classify the following user query as either:
    - "legal" → if it involves contracts, agreements, legal rights, court matters, legal advice, clauses, legal consequences, or anything law related
    - "general" → if it is a greeting, small talk, general knowledge, or anything not related to law

    Respond with ONLY one word: legal or general

    User query:{user_query}
    """
    
    response = llm.invoke(prompt)
    query_type = response.content.strip().lower
    
    if query_type not in ["legal", "general"]:
        query_type = "general"
        
    return {**state, "query_type":query_type}