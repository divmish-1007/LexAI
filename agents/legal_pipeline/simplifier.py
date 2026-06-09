import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import LegalState
from config.llm import llm

def simplifier_agent(state: LegalState) -> LegalState:
    clauses = state.get("clauses", [])
    document_type = state.get("document_type", "Unknown")

    messages = [
        SystemMessage(content="""You are a legal document simplifier who specializes in converting complex legal language into plain, simple English that anyone can understand.

        For each clause:
        1. Rewrite it in simple, clear English
        2. Keep the meaning intact — do not change what the clause says
        3. Use short sentences, avoid legal jargon
        4. Make it understandable for a person with no legal background

        Respond ONLY in this exact JSON format, no extra text, no markdown:
        {
            "simplified_clauses": [
                {
                    "original": "...",
                    "simplified": "..."
                }
            ]
        }"""),
        HumanMessage(content=f"""Document Type: {document_type}

        Here are the clauses to simplify:

        {json.dumps(clauses, indent=2)}""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "simplified_clauses": [
                {"original": clause, "simplified": clause}
                for clause in clauses
            ]
        }

    return {
        **state,
        "simplified_clauses": result.get("simplified_clauses", [])
    }