import fitz
import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import LegalState
from config.llm import llm

def reader_agent(state: LegalState) -> LegalState:
    raw_text = state.get("raw_text", "")
    
    if not raw_text:
        doc = fitz.open(state["pdf_path"])
        raw_text = "\n".join(page.get_text() for page in doc)
        doc.close()
   
    messages = [
        SystemMessage(content="""You are a legal document analyst.
        Given a legal document, you must:
        1. Identify the document type (e.g. Rent Agreement, Job Offer Letter, Loan Agreement, NDA)
        2. Split the document into individual meaningful clauses

        Respond ONLY in this exact JSON format, no extra text, no markdown:
        {
            "document_type": "...",
            "clauses": [
                "clause 1 text",
                "clause 2 text"
            ]
        }"""),
        HumanMessage(content=f"Here is the legal document:\n\n{raw_text}")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "document_type": "Unknown",
            "clauses": [raw_text]
        }

    return {
        **state,
        "raw_text": raw_text,
        "document_type": result.get("document_type", "Unknown"),
        "clauses": result.get("clauses", [raw_text])
    }