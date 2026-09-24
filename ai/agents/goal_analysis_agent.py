from langchain_groq import ChatGroq
from ai.state import FoodState
from ai.llm import llm
from ai.nutrition_targets import compute_targets

GOAL_ANALYSIS_SYSTEM_PROMPT = """You are a nutrition goal-analysis assistant. You are given a user's \
stated nutrition goal and their already-calculated daily nutrition totals (calories, protein, carbs, \
fat, sugar, and macro percentages). The totals were computed by another system from real food-database \
data — do NOT recalculate or second-guess them. If the user's PERSONAL DAILY TARGETS are provided, \
they were computed by a formula (Mifflin-St Jeor + activity level + goal): use exactly those numbers and do \
NOT invent different targets. If no targets are provided, give general advice only and do not invent \
numeric targets.

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

WHEN PERSONAL TARGETS ARE PROVIDED, be strict and specific:
- Compare today's intake with each target (calories, protein, carbs, fat) and state clearly how far over or \
under each one is, using the "remaining" numbers you are given.
- Base every ADD / REDUCE item on those remaining amounts (e.g. "you still have about 45 g protein and \
400 kcal left: a 150 g grilled chicken breast plus a large salad fits"). Give approximate grams.
- The user may have logged only part of their day. If intake is far below target, say the day may not be \
fully logged instead of alarming them, and describe what the REST of the day should look like.
- Never suggest going below the calorie target given, and mention any note attached to the targets.

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
    targets = compute_targets(state.get("profile"), goal)
    state["targets"] = targets or {}
    if not analyzer_output or (not goal and not targets):
        state["goal_analysis_content"] = ""
        return state
    totals = analyzer_output["totals"]
    macro_percentages = analyzer_output["macro_percentages"]
    targets_block = ""
    if targets:
        profile = state.get("profile") or {}
        remaining = {
            "calories": targets["target_kcal"] - totals["calories"],
            "protein": targets["protein_g"] - totals["protein_g"],
            "carbs": targets["carbs_g"] - totals["carbs_g"],
            "fat": targets["fat_g"] - totals["fat_g"],
        }
        notes = " ".join(targets.get("notes") or []) or "none"
        targets_block = f"""

PERSONAL DAILY TARGETS (computed by formula - use these exact numbers):
- Person: {profile.get('gender')}, {profile.get('age')} years, {profile.get('height_cm')} cm, \
{profile.get('weight_kg')} kg, {targets['activity_label']} (BMI {targets['bmi']})
- Goal type: {targets['goal_type']}
- Maintenance calories: {targets['maintenance_kcal']} kcal/day
- Target: {targets['target_kcal']} kcal, {targets['protein_g']} g protein, \
{targets['carbs_g']} g carbs, {targets['fat_g']} g fat
- Remaining today (target minus eaten; negative = over): {remaining['calories']:.0f} kcal, \
{remaining['protein']:.0f} g protein, {remaining['carbs']:.0f} g carbs, {remaining['fat']:.0f} g fat
- Notes: {notes}"""

    human_prompt = f"""User's goal: {goal or 'not stated'}

Today's totals (already calculated, do not recompute):
- Calories: {totals['calories']:.0f} kcal
- Protein: {totals['protein_g']:.1f} g
- Carbohydrates: {totals['carbs_g']:.1f} g
- Fat: {totals['fat_g']:.1f} g
- Sugar: {totals['sugar_g']:.1f} g

Macro split: {macro_percentages['protein_pct']}% protein, \
{macro_percentages['carbs_pct']}% carbs, {macro_percentages['fat_pct']}% fat{targets_block}

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
