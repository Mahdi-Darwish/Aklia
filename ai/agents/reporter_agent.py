import os
import json
from langchain_groq import ChatGroq
from ai.state import FoodState
MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

REPORTER_SYSTEM_PROMPT = """You are a nutrition reporting assistant. You will be given a JSON object \
with two keys:
- "nutrition_data": individual food items with their nutrition values, totals, and macro percentages.
- "goal_analysis": either null, or goal-based coaching text (what's going well, what to add/reduce, a \
  main recommendation) already written by another agent from the same totals.

Write a clean, professional daily nutrition report.

Guidelines:
- Start with a one-line summary (total calories, and a quick take like "balanced" or "carb-heavy").
- Include a markdown table of items (from nutrition_data.items) with calories, protein, carbs, fat, sugar.
- Include a totals row.
- Show macro percentage breakdown (protein/carbs/fat).
- Mention if any items couldn't be matched/estimated, if that's the case.
- If any item has "unit_conversion_estimated": true, note briefly that its gram conversion is a rough \
  approximation (USDA had no exact portion data for that food/unit combination) rather than presenting \
  it with the same confidence as exact conversions.
- If any item has "source": "ai_estimate", note briefly that its nutrition values are an AI estimate \
  (no USDA or Open Food Facts match was found for that food) rather than presenting it with the same \
  confidence as database-sourced values.
- Keep the numbers section factual and neutral — no health judgments or diet advice there, just report \
  the numbers.
- If "goal_analysis" is not null, add a final "Goal-Based Recommendations" section that presents that \
  text clearly (light reformatting for consistency is fine, but do not change its substance or invent \
  new advice). If "goal_analysis" is null, omit that section entirely — do not invent goal advice.

LANGUAGE RULE — IMPORTANT:
The final response MUST be written in the same language as the user's
original message.

If the user writes in Arabic, write the entire response in Arabic.
If the user writes in English, write the entire response in English.
If the user mixes languages, use the dominant language.

NEVER switch to English simply because the nutrition data, food names,
or internal agent outputs are in English.

The final response must be natural and easy for the user to understand."""

def generate_report(state: FoodState) -> FoodState:
    if not state.get("analyzer_output"):
        error_detail = state.get("error", "")
        if not state.get("parser_output"):
            state["report_content"] = (
                "I couldn't understand your food description well enough to analyze it. "
                "Could you try rephrasing it, listing each food item a bit more simply "
                "(e.g. \"2 eggs, 1 slice of toast, 1 can of soda\")?"
            )
        else:
            state["report_content"] = (
                "I understood your food description, but couldn't fetch nutrition data "
                "due to a backend issue (nutrition database temporarily unavailable). "
                "This isn't a problem with your input — please try again in a moment."
                + (f"\n\n(Technical detail: {error_detail})" if error_detail else "")
            )
        return state
    llm = ChatGroq(model=MODEL_NAME, temperature=0.3, max_tokens=1000)
    report_input = {
        "nutrition_data": state["analyzer_output"],
        "goal_analysis": state.get("goal_analysis_content") or None,}
    messages = [
        ("system", REPORTER_SYSTEM_PROMPT),
        ("human", json.dumps(report_input, indent=2)),]
    response = llm.invoke(messages)
    state["report_content"] = response.content.strip()
    return state