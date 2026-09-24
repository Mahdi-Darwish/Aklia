import os
from ai.llm import llm
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from ai.state import FoodState
Unit = Literal["g", "kg", "ml", "tsp", "tbsp", "cup", "piece", "slice", "can", "bottle", "pack"]

class FoodItem(BaseModel):
    name: str = Field(description="Normalized food name, e.g. 'banana', 'bread, toasted'")
    quantity: float = Field(description="Numeric quantity, e.g. 1, 2, 0.5")
    unit: Unit = Field(description="Unit of measurement for the quantity")
    estimated_grams: float = Field(
        description=(
            "Your best estimate of the TOTAL weight in grams for this quantity of this "
            "specific food, using general culinary knowledge. E.g. '1 slice of pizza' -> "
            "~120g, '1 slice of bread' -> ~30g, '1 cup of cooked rice' -> ~160g, "
            "'1 tbsp of olive oil' -> ~14g, '1 can of soda' -> ~355g, '1 sandwich' -> ~200g. "
            "This is used as a fallback ONLY when the unit is not already 'g'/'kg' and the "
            "food isn't one of the standard countable produce items (apple, banana, egg, "
            "orange, pear, peach, kiwi, date, tomato, avocado, potato, onion) that the "
            "analyzer already has a fixed reference weight for — so make it as realistic as "
            "possible for THIS food, not a generic guess for the unit alone."
        )
    )


class ParsedMeals(BaseModel):
    items: List[FoodItem] = Field(description="All individual food items extracted from the input")

PARSER_SYSTEM_PROMPT = """You are a food-log parser. Given a user's description of what they ate, \
extract a clean list of individual food items with an estimated quantity and unit.

Rules:
- ALWAYS break a homemade/prepared dish into its base component + topping/filling/spread, even if the \
  user names it as one thing. Nutrition databases contain raw ingredients and packaged retail products — \
  they do NOT contain entries for someone's homemade combination dish. A dish name is a strong signal to \
  decompose it, not a food to search for as-is.
  Example: "one nutella crepe" -> TWO items: {"name": "crepe", ...} AND {"name": "nutella (hazelnut spread)", ...} \
  — NOT one item called "nutella crepe" (that exact phrase will never be found in any nutrition database).
  Example: "toast with butter" -> "bread, toasted" + "butter".
  Example: "a cheese sandwich" -> "bread, slice" (x2) + "cheese, slice".
- EXCEPTION — pizza is ONE item, whatever the toppings: "2 pepperoni pizza slices" -> a single item \
{"name": "pizza", "quantity": 2, "unit": "slice", ...}. NEVER list its toppings (pepperoni, cheese, \
mushrooms...) as separate items — the database entry for pizza already includes them, so adding them \
double-counts calories. In general, never invent an item the user did not mention.
- A branded, single-ingredient product (e.g. "a can of Pepsi", "a Snickers bar") does NOT need decomposing — \
  keep those as one item, since packaged-product databases DO contain those directly.

FOOD NAME RULES (the name is used to search the USDA database, so keep it clean and consistent):
- name = the plain, singular, lowercase food, e.g. "rice", "chicken breast", "apple", "pasta".
- Add ", raw" ONLY if the user explicitly said raw / uncooked / dry (e.g. "dry pasta" -> "pasta, raw").
- Add a cooking method (grilled, boiled, fried, baked) ONLY if the user said it (e.g. "grilled chicken" -> \
"chicken, grilled"). Otherwise add NO extra descriptors — never invent "cooked", "white", "fresh", "steamed".
- Always use the same name for the same food ("rice", never sometimes "white rice" / "steamed rice").

WEIGHT GIVEN EXPLICITLY — ALWAYS PRESERVE IT EXACTLY:
If the user states an explicit weight ("200g grilled chicken", "150 grams of rice", "0.5 kg beef", \
"150g apple"), use unit="g" (or "kg") with that EXACT quantity. Do NOT convert it to "piece" or any \
other unit, do NOT round it, and do NOT second-guess it with your own estimate — the user's stated \
weight is ground truth and takes priority over everything below.

COUNTABLE FOODS WITH NO STATED WEIGHT — use unit="piece", do NOT invent grams:
For naturally countable foods with no explicit weight — apple, banana, egg, orange, pear, peach, kiwi, \
date, tomato, avocado, potato, onion — use unit="piece" and the given count as quantity (e.g. "2 apples" \
-> quantity=2, unit="piece"). The analyzer has a fixed standard reference weight (medium size, raw) for \
each of these, so do NOT estimate grams for them yourself — just get the name and count right.

EVERYTHING ELSE WITH NO STATED WEIGHT:
For any other food with no explicit weight (rice, pasta, meat, bread, cheese, sauces, snacks, drinks, \
decomposed dish components, etc.), pick whichever of these units is most natural — g, ml, tsp, tbsp, \
cup, piece, slice, can, bottle, pack — and provide your own realistic estimated_grams for that specific \
food and quantity, using general culinary knowledge of typical portion sizes. If no quantity is given, \
make a reasonable default assumption (e.g. "toast" -> 1 slice).
- For estimated_grams, think about the SPECIFIC food, not just the unit. A "slice" of pizza weighs much \
  more than a "slice" of bread or a "slice" of cheese — use realistic, food-specific gram weights.

LANGUAGE RULE:
Preserve the language of the user's input when producing any text fields. Do not translate the user's \
food descriptions unless necessary for standardized food identification.
"""

MAX_PARSER_RETRIES = 2

def parse_food_input(state: FoodState) -> FoodState:
    state["parser_retry_count"] = state.get("parser_retry_count", 0) + 1
    structured_llm = llm.with_structured_output(ParsedMeals)

    messages = [
        ("system", PARSER_SYSTEM_PROMPT),
        ("human", state["meals"]),
    ]
    try:
        result: ParsedMeals = structured_llm.invoke(messages)
    except Exception as exc:
        error_msg = f"Parser failed to produce structured output: {exc}"
        print(f"[parser_node] {error_msg}")
        state["error"] = error_msg
        state["parser_output"] = []
        return state
    state["parser_output"] = [item.model_dump() for item in result.items]
    state["error"] = None
    return state