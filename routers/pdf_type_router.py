from graph.state import LegalState
from config.llm import llm
import fitz

def pdf_type_router(state: LegalState):
    pdf_path = state.get("pdf_path", "")

    doc = fitz.open(pdf_path)
    raw_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    
    prompt = f""" You are a document classifier.
    Read the following text extracted from a PDF and classify it as either:
    - "legal" → if it is a contract, agreement, NDA, terms of service, legal notice, court document, or any legal document
    - "normal" → if it is a research paper, textbook, manual, report, or any non-legal document

    Respond with ONLY one word: legal or normal
    
    Document text (first 1000 characters):
    {raw_text[:1000]}
    """
    
    response = llm.invoke(prompt)
    pdf_type = response.content.strip().lower()
    
    if pdf_type not in ["normal", "legal"]:
        pdf_type = "normal"
    
    return {
        **state,
        "raw_text": raw_text,   
        "pdf_type":pdf_type
    }