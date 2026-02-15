import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from base_agent import AgentState
from sale_agent import SaleAgent
from analysis_agent import AnalysisAgent
from orchestration_agent import create_supervisor_node, members

# Load environment variables
load_dotenv()

# Initialize agents
sale_agent = SaleAgent()
analysis_agent = AnalysisAgent()

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Supervisor", create_supervisor_node)
workflow.add_node("Sales_Agent", sale_agent.get_node())
workflow.add_node("Analysis_Agent", analysis_agent.get_node())

# Entry point
workflow.set_entry_point("Supervisor")

# Normal edges from agents back to Supervisor
for member in members:
    workflow.add_edge(member, "Supervisor")

# Conditional edges from Supervisor to agents or END
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Sales_Agent": "Sales_Agent",
        "Analysis_Agent": "Analysis_Agent",
        "FINISH": END
    }
)

app = workflow.compile()

def run_chat():
    print("Welcome to the Sales Multi-Agent System. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                        
            final_state = None
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "sender": "User",
            }
            
            # To maintain history across turns without a DB, we can use a MemorySaver?
            # LangGraph has MemorySaver.
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            
            # Recompile with checkpointer
            app_with_memory = workflow.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": "1"}}
            
            for event in app_with_memory.stream(inputs, config=config):
                for key, value in event.items():
                    if key != "Supervisor":
                         if "messages" in value and len(value["messages"]) > 0:
                            last_msg = value["messages"][-1]
                            sender = value.get("sender", key)
                            if isinstance(last_msg, BaseMessage):
                                 print(f"\n[{sender}]: {last_msg.content}")
                                 
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_chat()
