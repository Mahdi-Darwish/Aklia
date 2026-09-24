from dotenv import load_dotenv
load_dotenv()
from ai.graph import build_graph
from ai.state import FoodState
def run(goal: str, meals: str) -> str:
    app = build_graph()
    initial_state: FoodState = {
        "goal": goal,
        "meals": meals,
        "parser_output": [],
        "analyzer_output": {},
        "goal_analysis_content": "",
        "report_content": "",
        "retry_count": 0,
        "parser_retry_count":0,
        "error": None,}
    final_state = app.invoke(initial_state)
    return final_state["report_content"]

if __name__ == "__main__":
    goal = input("Enter your goal: ")
    food = input("What did you eat today? ")
    report = run(goal, food)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)