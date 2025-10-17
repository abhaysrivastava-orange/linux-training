from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Define the agent's state ---
class AgentState:
    question: str
    answer: str

# --- Step 1: Call LLM ---
def answer_question(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = llm.invoke(f"Answer this question clearly:\n\n{state.question}")
    state.answer = response.content
    return state

# --- Build LangGraph ---
graph = StateGraph(AgentState)
graph.add_node("qa", answer_question)
graph.set_entry_point("qa")
graph.add_edge("qa", END)

# --- Compile and export graph for LangGraph SaaS ---
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
