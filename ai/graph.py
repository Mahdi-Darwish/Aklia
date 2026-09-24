from typing import Literal
from langgraph.graph import START,END,StateGraph
from ai.state import FoodState
from ai.agents.parser_agent import MAX_PARSER_RETRIES, parse_food_input
from ai.agents.analyzer_agent import analyze_nutrition
from ai.agents.reporter_agent import generate_report
from ai.agents.goal_analysis_agent import analyze_goal
MAX_RETRIES = 2

def is_parsed(state:FoodState) ->Literal["anayzer_node","parser_node","reporter_node"]:
    if not state.get("error"):
        return "analyzer_node"
    if state.get("parser_retry_count",0)<MAX_PARSER_RETRIES:
        return "parser_node"
    return "reporter_node"
        
def is_valid(state:FoodState) ->Literal["goal_analysis_node","analyzer_node"]:
    if state.get("error") and state.get("retry_count",0) <MAX_RETRIES:
        return "analyzer_node"
    return "goal_analysis_node"
def build_graph():
    agent_builder  = StateGraph(FoodState)
    agent_builder.add_node("parser_node",parse_food_input)
    agent_builder.add_node("analyzer_node",analyze_nutrition)
    agent_builder.add_node("goal_analysis_node",analyze_goal)
    agent_builder.add_node("reporter_node",generate_report)

    agent_builder.add_edge(START,"parser_node")
    agent_builder.add_conditional_edges("parser_node", is_parsed, ["analyzer_node", "parser_node", "reporter_node"])
    agent_builder.add_conditional_edges("analyzer_node",is_valid,["goal_analysis_node","analyzer_node"],)
    agent_builder.add_edge("goal_analysis_node","reporter_node")
    agent_builder.add_edge("reporter_node",END)
    agent = agent_builder.compile()
    return agent
