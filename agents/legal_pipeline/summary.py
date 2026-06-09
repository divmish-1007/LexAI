import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import LegalState
from config.llm import llm

def summary_agent(state: LegalState) -> LegalState:
    document_type = state.get("document_type", "Unknown")
    risky_clauses = state.get("risky_clauses", [])
    simplified_clauses = state.get("simplified_clauses", [])

    messages = [
        SystemMessage(content="""You are a senior legal advisor preparing a comprehensive document analysis report.

        Your report must include:
        1. Document Overview — what this document is and its purpose
        2. Key Terms — most important obligations and rights of each party
        3. Risk Assessment — summary of all risks found, grouped by severity
        4. Simplified Summary — what this document means in plain English
        5. Recommendations — what the signing party should be aware of or negotiate

        Write in a professional yet clear tone. Use sections with headers.
        The report should be useful for someone who has never read the document."""),
        
        HumanMessage(content=f"""Document Type: {document_type}

        Risky Clauses Found:
        {json.dumps(risky_clauses, indent=2)}

        Simplified Clauses:
        {json.dumps(simplified_clauses, indent=2)}

        Please prepare a comprehensive legal analysis report.""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "final_summary": response.content.strip()
    }