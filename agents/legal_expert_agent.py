from graph.state import LegalState
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm import llm

def legal_expert_agent(state: LegalState):
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    
    messages = [
        SystemMessage(content=""" You are LexAI, a senior legal expert with decades of experience across contract law, civil law, criminal law, property law, employment law, and corporate law.
        When answering:
        1. Provide accurate, detailed legal information
        2. Cite relevant legal principles or concepts where applicable
        3. Explain implications and consequences clearly
        4. If jurisdiction matters, mention it and ask for clarification if needed
        5. Always recommend consulting a licensed attorney for official legal advice
        6. Use professional legal tone but remain understandable
        
        You are not just a chatbot — you are a knowledgeable legal advisor."""),
        
        *chat_history, 
        
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    updated_history = chat_history + [
        HumanMessage(content=user_query), 
        response
    ]
    
    return {
        **state, 
        "agent_response": response.content.strip(),
        "chat_history": updated_history
    }