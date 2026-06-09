from langchain_core.messages import HumanMessage, SystemMessage
from config.llm import llm
from graph.state import LegalState

def general_chat_agent(state:LegalState):
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    
    messages = [
        SystemMessage(content="""You are LexAI, a specialized Legal AI Assistant.

        Your expertise is strictly limited to:
        1. Legal document analysis and risk detection
        2. Legal questions and expert legal advice
        3. Document based question answering
        4. Legal concepts, rights, and obligations

        If the user asks anything outside these areas:
        - Politely but firmly let them know it is outside your scope
        - Remind them what you can help with
        - Redirect them toward legal topics

        If the user greets you or introduces themselves:
        - Respond warmly and professionally
        - Introduce yourself as LexAI
        - Tell them what you can help with

        If the user asks who you are or what you do:
        - Explain you are a Legal AI Assistant
        - List your capabilities clearly

        Never answer questions about cooking, entertainment, sports, technology, 
        science, or any non-legal topic. Always redirect back to legal matters."""),
        *chat_history,
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    updated_history = chat_history + [
        HumanMessage(content=user_query),
        response
    ]
    
    return{
        **state, 
        "agent_response": response.content.strip(),
        "chat_history": updated_history
    }