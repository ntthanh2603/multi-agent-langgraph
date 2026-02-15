from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from base_agent import create_agent
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import functools
import operator
import os

# Mock database
PRODUCTS = {
    "iphone 15": {"price": 999, "stock": 50},
    "macbook pro": {"price": 1999, "stock": 20},
    "ipad air": {"price": 599, "stock": 100},
    "airpods max": {"price": 549, "stock": 10},
}

@tool
def check_stock(product_name: str) -> str:
    """Check if a product is in stock."""
    product = PRODUCTS.get(product_name.lower())
    if product:
        return f"We have {product['stock']} units of {product_name} in stock."
    return f"Sorry, we don't carry {product_name}."

@tool
def get_price(product_name: str) -> str:
    """Get the price of a product."""
    product = PRODUCTS.get(product_name.lower())
    if product:
        return f"The price of {product_name} is ${product['price']}."
    return f"Sorry, price not found for {product_name}."

@tool
def list_products() -> str:
    """List all available products."""
    return ", ".join(PRODUCTS.keys())

class SaleAgent:
    def __init__(self):
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        api_base = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(model=model_name, temperature=0, base_url=api_base)
        self.tools = [check_stock, get_price, list_products]
        self.agent = create_agent(
            self.llm,
            self.tools,
            system_message="You are a Sales AI. Your job is to help customers find products, check prices, and availability.",
            name="Sales_Agent"
        )
    
    def get_node(self):
        # We invoke the graph with the current state.
        # Since 'agent' is a CompiledGraph, calling it returns the final state.
        # But LangGraph nodes expect to receive state and return an update.
        # create_agent returns a CompiledGraph.
        # A CompiledGraph IS a valid node in LangGraph if inputs/outputs match.
        return self.agent
