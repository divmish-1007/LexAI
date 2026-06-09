from graph.state import LegalState

def input_router(state: LegalState):
    if(state.get("pdf_path")):
        return {**state, "input_type": "pdf"}
    else:
        return {**state, "input_type": "query"}