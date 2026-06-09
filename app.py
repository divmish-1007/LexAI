import chainlit as cl
from graph.workflow import lexai_graph
from graph.state import LegalState

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])
    await cl.Message(content="👋 Welcome to **LexAI** — Your Legal Intelligence Assistant!\n\nI can help you with:\n\n📄 **Legal Document Analysis** — Upload a legal PDF for risk detection, simplification, and full report\n\n📝 **Document Q&A** — Upload any PDF and ask questions about it\n\n⚖️ **Legal Advice** — Ask any legal question and get expert guidance\n\n💬 **General Chat** — Ask me anything!\n\n**To get started:** Type a message or upload a PDF below ⬇️").send()

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history", [])

    # --- Build initial state ---
    state: LegalState = {
        "pdf_path": "",
        "user_query": message.content or "",
        "input_type": "",
        "pdf_type": "",
        "query_type": "",
        "raw_text": "",
        "document_type": "",
        "clauses": [],
        "risky_clauses": [],
        "simplified_clauses": [],
        "final_summary": "",
        "chat_history": chat_history,
        "agent_response": ""
    }

    # --- Handle PDF upload ---
    if message.elements:
        pdf_file = message.elements[0]
        state["pdf_path"] = pdf_file.path
        state["user_query"] = message.content or "Analyze this document"

        # Show processing message
        processing_msg = cl.Message(content="🔍 Analyzing your document, please wait...")
        await processing_msg.send()

    # --- Invoke graph ---
    try:
        result = lexai_graph.invoke(state)

        # Update chat history for next turn
        cl.user_session.set("chat_history", result.get("chat_history", []))

        # --- Display results ---
        if result.get("final_summary"):
            await display_legal_analysis(result)
        else:
            await cl.Message(content=result.get("agent_response", "")).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Something went wrong: {str(e)}").send()


async def display_legal_analysis(result: dict):
    document_type = result.get("document_type", "Unknown")
    risky_clauses = result.get("risky_clauses", [])
    simplified_clauses = result.get("simplified_clauses", [])
    final_summary = result.get("final_summary", "")

    # --- Document Type ---
    await cl.Message(content=f"## 📄 Document Type\n**{document_type}**").send()

    # --- Risk Analysis ---
    if risky_clauses:
        risk_text = "## ⚠️ Risky Clauses Found\n\n"
        for i, risk in enumerate(risky_clauses, 1):
            severity = risk.get("severity", "unknown").upper()
            emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            risk_text += f"{emoji} **Risk {i} — {severity}**\n"
            risk_text += f"**Clause:** {risk.get('clause', '')}\n"
            risk_text += f"**Reason:** {risk.get('reason', '')}\n\n"
        await cl.Message(content=risk_text).send()
    else:
        await cl.Message(content="✅ **No risky clauses found!**").send()

    # --- Simplified Clauses ---
    if simplified_clauses:
        simplified_text = "## 📝 Simplified Clauses\n\n"
        for i, item in enumerate(simplified_clauses, 1):
            simplified_text += f"**Clause {i}:**\n"
            simplified_text += f"- 📜 **Original:** {item.get('original', '')}\n"
            simplified_text += f"- ✅ **Simplified:** {item.get('simplified', '')}\n\n"
        await cl.Message(content=simplified_text).send()

    # --- Final Summary ---
    await cl.Message(content=f"## 📊 Legal Analysis Report\n\n{final_summary}").send()

    # --- Follow up ---
    await cl.Message(content="💬 **Feel free to ask any questions about this document!**").send()