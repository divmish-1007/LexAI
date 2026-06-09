import json
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import LegalState
from config.llm import llm

def risk_detector_agent(state: LegalState) -> LegalState:
    clauses = state.get("clauses", [])

    messages = [
        SystemMessage(content="""You are a legal risk analyst with expertise in identifying problematic clauses in legal documents.

        Analyze the given clauses and identify which ones are risky, unfair, or potentially harmful to the signing party.

        For each risky clause provide:
        1. The clause text
        2. The reason it is risky
        3. Severity level: "high", "medium", or "low"

        Respond ONLY in this exact JSON format, no extra text, no markdown:
        {
            "risky_clauses": [
                {
                    "clause": "...",
                    "reason": "...",
                    "severity": "high or medium or low"
                }
            ]
        }

        If no risky clauses are found, return:
        {
            "risky_clauses": []
        }"""),
        HumanMessage(content=f"""Document Type: {state.get("document_type", "Unknown")}

        Here are the clauses to analyze:

        {json.dumps(clauses, indent=2)}""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"risky_clauses": []}

    return {
        **state,
        "risky_clauses": result.get("risky_clauses", [])
    }