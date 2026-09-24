"""
Live sanity check of the USDA matching - run this BEFORE the deadline.

    python check_foods.py                # checks a default list of common foods
    python check_foods.py "rice" "banana, raw" "chicken, grilled"

For every food it prints which USDA entry was chosen and its kcal/100g, so you
can eyeball that rice ~130, apple ~52, grilled chicken breast ~165, etc.
(No LLM calls - only the USDA lookup + the new validation logic.)
"""
import sys
from dotenv import load_dotenv
load_dotenv()

from ai.agents import analyzer_agent

DEFAULT_FOODS = [
    "rice", "rice, raw", "rice, brown", "pasta", "apple", "banana", "orange",
    "chicken, grilled", "chicken breast", "egg", "salmon", "potato", "broccoli",
    "bread", "milk", "yogurt", "oats", "olive oil", "almond", "cheese",
]

# rough sanity ranges (kcal per 100 g) for the defaults - flags anything wildly off
EXPECTED = {
    "rice": (110, 150), "rice, raw": (340, 380), "rice, brown": (105, 140), "pasta": (130, 175),
    "apple": (45, 65), "banana": (80, 95), "orange": (40, 55), "chicken, grilled": (150, 190),
    "chicken breast": (150, 190), "egg": (130, 200), "salmon": (120, 230), "potato": (70, 110),
    "broccoli": (28, 40),
}

foods = sys.argv[1:] or DEFAULT_FOODS
for name in foods:
    try:
        ranked = analyzer_agent._lookup_usda(name)
    except analyzer_agent.USDALookupError as exc:
        print(f"ERROR  {name:18s} {exc}")
        continue
    chosen = None
    for food in ranked[:8]:
        c = analyzer_agent._extract_usda_nutrients_per_100g(food)
        if analyzer_agent._has_valid_calories(c) and analyzer_agent._is_plausible(c):
            chosen, cal = food, c["calories"]
            break
    if chosen is None:
        print(f"MISS   {name:18s} no plausible USDA entry -> would fall through to Open Food Facts / AI estimate")
        continue
    lo, hi = EXPECTED.get(name, (0, 950))
    flag = "OK   " if lo <= cal <= hi else "CHECK"
    print(f"{flag}  {name:18s} {cal:6.0f} kcal/100g  <- {chosen['description'][:70]}")
