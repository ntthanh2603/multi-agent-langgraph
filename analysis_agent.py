from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from base_agent import create_agent
from typing import TypedDict, Annotated, Sequence
import functools
import json
import os

@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """Calculate the final price after applying a discount."""
    if discount_percent < 0 or discount_percent > 100:
        return "Invalid discount percentage. Must be between 0 and 100."
    final_price = price * (1 - discount_percent / 100)
    return f"The discounted price is ${final_price:.2f}."

@tool
def calculate_roi(investment: float, return_amount: float) -> str:
    """Calculate Return on Investment (ROI)."""
    if investment == 0:
        return "Investment cannot be zero."
    roi = ((return_amount - investment) / investment) * 100
    return f"The ROI is {roi:.2f}%."

@tool
def analyze_sales_trend(sales_data: str) -> str:
    """Analyze sales trend from a JSON string of sales figures over time.
    Format: {"Jan": 100, "Feb": 200, ...}
    Returns a summary of the trend.
    """
    try:
        data = json.loads(sales_data)
        months = list(data.keys())
        values = list(data.values())
        
        if len(values) < 2:
            return "Not enough data to analyze trend."
            
        trend = "increasing" if values[-1] > values[0] else "decreasing"
        average = sum(values) / len(values)
        
        return f"Sales trend is {trend}. Average sales per month: {average:.2f} units. Highest: {max(values)} in {months[values.index(max(values))]}."
    except json.JSONDecodeError:
        return "Error: Input must be a valid JSON string representing sales data."

class AnalysisAgent:
    def __init__(self):
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        api_base = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(model=model_name, temperature=0, base_url=api_base)
        
        self.tools = [calculate_discount, calculate_roi, analyze_sales_trend]
        self.agent = create_agent(
            self.llm,
            self.tools,
            system_message="You are an Analysis AI. You help with calculations, financial analysis, and identifying trends in sales data.",
            name="Analysis_Agent"
        )
    
    def get_node(self):
        return self.agent
