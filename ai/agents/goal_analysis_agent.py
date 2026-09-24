
from langchain_groq import ChatGroq
from ai.state import FoodState
from ai.llm import llm

GOAL_ANALYSIS_SYSTEM_PROMPT = """You are a nutrition goal-analysis assistant. You are given a user's \
stated nutrition goal and their already-calculated daily nutrition totals (calories, protein, carbs, \
fat, sugar, and macro percentages). The totals were computed by another system from real food-database \
data — do NOT recalculate or second-guess them, and do NOT invent numeric targets that weren't given \
to you.

Your job:
1. Judge how today's intake fits the stated goal, using general nutrition knowledge (e.g. muscle gain \
   generally wants higher protein and a calorie surplus; fat loss/cutting generally wants a calorie \
   deficit with enough protein to preserve muscle; bulking wants a calorie surplus).
2. Call out what's going well.
3. Call out what needs improvement.
4. Recommend specific foods or nutrients to ADD.
5. Recommend specific foods or nutrients to REDUCE, with a rough practical sense of how much \
   (e.g. "cut the rice portion by about a third") rather than vague advice.
6. Give one clear, practical MAIN RECOMMENDATION.

Keep it concise and practical. Do not diagnose medical conditions or claim this is medical advice.

Return your answer in this structure:

Goal Analysis
- what's going well
- what needs improvement

ADD
- foods/nutrients to consider adding

REDUCE
- foods/nutrients to consider reducing

MAIN RECOMMENDATION
- one practical recommendation


LANGUAGE RULE:
If you produce explanatory text, use the same language as the user's
original input.
Do not switch languages.
"""


def analyze_goal(state: FoodState) -> FoodState:
    goal = (state.get("goal") or "").strip()
    analyzer_output = state.get("analyzer_output")

    # No goal given, or the analyzer never produced totals (e.g. it's on
    # its way to the reporter's error-fallback path) — nothing to analyze.
    if not goal or not analyzer_output:
        state["goal_analysis_content"] = ""
        return state

    totals = analyzer_output["totals"]
    macro_percentages = analyzer_output["macro_percentages"]

    human_prompt = f"""User's goal: {goal}

Today's totals (already calculated, do not recompute):
- Calories: {totals['calories']:.0f} kcal
- Protein: {totals['protein_g']:.1f} g
- Carbohydrates: {totals['carbs_g']:.1f} g
- Fat: {totals['fat_g']:.1f} g
- Sugar: {totals['sugar_g']:.1f} g

Macro split: {macro_percentages['protein_pct']}% protein, \
{macro_percentages['carbs_pct']}% carbs, {macro_percentages['fat_pct']}% fat

Analyze this intake against the stated goal and respond in the structure you were given."""

    messages = [
        ("system", GOAL_ANALYSIS_SYSTEM_PROMPT),
        ("human", human_prompt),
    ]

    try:
        response = llm.invoke(messages)
        state["goal_analysis_content"] = response.content.strip()
    except Exception as exc:
        print(f"[goal_analysis_node] {exc}")
        state["goal_analysis_content"] = ""

    return state
