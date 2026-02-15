from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Define the members of the team
members = ["Sales_Agent", "Analysis_Agent"]

# Define the routing logic
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

class RouteResponse(BaseModel):
    next: Literal["Sales_Agent", "Analysis_Agent", "FINISH"]

options = ["FINISH"] + members

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

import os
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
api_base = os.getenv("OPENAI_API_BASE")
llm = ChatOpenAI(model=model_name, base_url=api_base)

def create_supervisor_node(state):
    """
    Supervisor node that decides the next step based on the conversation state.
    """
    # Create chain that outputs structured response
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    
    # Invoke chain with current state
    result = supervisor_chain.invoke(state)
    
    # Return the next step to update the state
    return {"next": result.next}
