from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import LegalState
from config.llm import llm

def pdf_qa_agent(state: LegalState) -> LegalState:
    user_query = state.get("user_query", "")
    raw_text = state.get("raw_text", "")
    chat_history = state.get("chat_history", [])
    document_type = state.get("document_type", "Unknown")

    messages = [
        SystemMessage(content=f"""You are LexAI, an intelligent document assistant.

        You have been given a document to analyze and answer questions about.
        Document Type: {document_type}

        Rules:
        1. Answer ONLY from the document content provided
        2. If the answer is not in the document, say "I could not find that information in the document"
        3. Quote relevant parts of the document when helpful
        4. Be precise and concise in your answers
        5. If the user asks for a summary, provide a clean structured overview
        6. Never make up information that is not in the document

        Document Content:
        {raw_text}"""),
        
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