# USDA key (free): https://fdc.nal.usda.gov/api-key-signup.html
import os
import requests
from pydantic import BaseModel, Field
from ai.llm import llm
from ai.state import FoodState
from ai.food_cache import get_cached_match, save_confirmed_match
import re
from concurrent.futures import ThreadPoolExecutor
USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")  # DEMO_KEY works but is rate-limited
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_DETAIL_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_HEADERS = {"User-Agent": "PrexilityMeal-NutritionAgent/1.0 (contact: set-a-real-contact-here)"}
NUTRIENT_IDS = { "calories": 1008,"protein_g": 1003,"carbs_g": 1005,"fat_g": 1004,"sugar_g": 2000,}  # USDA nutrient IDs
OFF_NUTRIENT_FIELDS = {  # Open Food Facts nutriment field names (per 100g)
    "calories": "energy-kcal_100g",
    "protein_g": "proteins_100g",
    "carbs_g": "carbohydrates_100g",
    "fat_g": "fat_100g",
    "sugar_g": "sugars_100g",
}
GENERIC_GRAMS_PER_UNIT = {
    "ml": 1,
    "tsp": 5,
    "tbsp": 15,
    "cup": 240,
    "piece": 100,
    "slice": 30,
    "can": 355,
    "bottle": 500,
    "pack": 50,
}
COUNT_BASED_FOODS = {
    "apple": 182,
    "banana": 118,
    "egg": 50,
    "orange": 131,
    "pear": 178,
    "peach": 150,
    "kiwi": 76,
    "date": 8,  
    "tomato": 123,
    "avocado": 150,
    "potato": 173,
    "onion": 110,
}
def _get_count_based_grams(name: str) -> float | None:
    """Fixed reference weight for a plain countable food (\"apple\",
    \"tomatoes\", \"green apple\", \"egg, boiled\"). Looks at the primary
    segment (before any comma), tries the whole phrase and then its last
    word (so \"green apple\" -> \"apple\"), each de-pluralized."""
    primary = (name or "").split(",")[0].strip().lower()
    if not primary:
        return None
    phrases = [primary]
    if " " in primary:
        phrases.append(primary.split()[-1])
    for word in phrases:
        candidates = [word]
        if word.endswith("ies") and len(word) > 3:
            candidates.append(word[:-3] + "y")
        if word.endswith("es") and len(word) > 2:
            candidates.append(word[:-2])  # tomatoes -> tomato, peaches -> peach
        if word.endswith("s") and len(word) > 1:
            candidates.append(word[:-1])  # bananas -> banana
        for candidate in candidates:
            if candidate in COUNT_BASED_FOODS:
                return COUNT_BASED_FOODS[candidate]
    return None

class USDALookupError(Exception):
    """Raised for transient/system failures in the PRIMARY (USDA) lookup —
    network error, timeout, bad response, missing key. NOT raised when USDA
    simply has no match (that's a valid outcome that triggers the OFF
    fallback, not a retry)."""

def _norm_word(word: str) -> str:
    word = word.strip().lower()
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("oes") and len(word) > 3:
        return word[:-2]  # tomatoes -> tomato, potatoes -> potato
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word
COOKED_BY_DEFAULT = {_norm_word(w) for w in (
    "rice", "pasta", "spaghetti", "macaroni", "noodle", "quinoa", "couscous", "bulgur",
    "lentil", "bean", "chickpea", "chicken", "turkey", "beef", "pork", "lamb", "veal",
    "fish", "salmon", "cod", "tilapia", "shrimp", "egg", "potato", "steak",
)}
COOKING_WORDS = {
    "cooked", "boiled", "baked", "roasted", "broiled", "grilled", "fried", "steamed",
    "pan-broiled", "pan-fried", "braised", "stewed", "poached", "simmered", "sauteed", "seared",
}
RAW_WORDS = {"raw", "uncooked", "dry"}

PROCESSING_PENALTY_KEYWORDS = (
    "dried", "dehydrated", "canned", "juice", "sauce", "syrup", "candied",
    "sweetened", "sulfured", "concentrate", "chips", "powder", "pie",
    "jam", "jelly", "paste", "smoked", "cured", "pickled",
    "frozen", "french", "mashed", "flakes", "hash", "puffed", "snack",
)
GENERIC_SEGMENT_WORDS = {
    "raw", "wheat", "white", "whole-wheat", "whole wheat", "plain", "original",
    "unsweetened", "cooked", "whole",
}
NON_DEFAULT_KEYWORDS = ("parboiled", "instant", "glutinous", "precooked", "imitation")
VARIETY_WORDS = {
    "fuji", "gala", "golden", "delicious", "granny", "smith", "honeycrisp", "braeburn",
    "mcintosh", "cortland", "empire", "navel", "valencia", "cara", "blood", "clementine",
}
CATEGORY_HEADS = {"fish", "crustacean", "mollusk", "cereal", "nut", "seed"}

DEFAULT_CUT_BONUS = {
    "chicken": (("breast", 40), ("meat only", 10)),
    "turkey": (("breast", 40), ("meat only", 10)),
    "beef": (("ground", 30),),
    "pork": (("loin", 20),),
    "egg": (("whole", 30), ("hard-boiled", 10)),
    "salmon": (("atlantic", 20), ("farmed", 10)),
    "apple": (("with skin", 5),),
    "cheese": (("cheddar", 150),),
    "rice": (("long-grain", 20), ("regular", 10)),
}


def _desc_words(description: str) -> set[str]:
    return {_norm_word(w) for w in re.split(r"[,\s()/]+", description.lower()) if w}

def _rank_usda_matches(query: str, foods: list[dict], qualifier_hint: str | None = None) -> list[dict]:
    """Returns ALL candidates that are really the requested food, best first.
    Deterministic: ties are broken by (shorter description, lower fdcId), so the
    same input always yields the same food. Empty list = USDA has no such food."""
    query_words = [_norm_word(w) for w in query.strip().split()]
    if not query_words:
        return []
    head_word, qualifier_words = query_words[0], query_words[1:]

    if qualifier_hint:
        for w in re.split(r"[,\s]+", qualifier_hint):
            w = _norm_word(w)
            if w and w != head_word and w not in qualifier_words:
                qualifier_words.append(w)

    explicit_raw = any(w in RAW_WORDS for w in qualifier_words)
    explicit_cooked = any(w in COOKING_WORDS for w in qualifier_words)
    if explicit_raw:
        state_pref = "raw"
    elif explicit_cooked or head_word in COOKED_BY_DEFAULT:
        state_pref = "cooked"
    else:
        state_pref = None 
    matches = []
    for food in foods:
        segs = [s.strip() for s in food.get("description", "").split(",")]
        segment_words = segs[0].split()
        if not segment_words:
            continue
        first = _norm_word(segment_words[0])
        desc_all = _desc_words(food.get("description", ""))
        all_words_present = all(w in desc_all for w in query_words)
        if first == head_word:
            if len(query_words) > 1 and not all_words_present:
                continue 
            extra = [_norm_word(w) for w in segment_words[1:]]
            if extra and not all(w in query_words for w in extra):
                continue
        elif len(query_words) > 1 and first in query_words and all_words_present:
            pass 
        elif first in CATEGORY_HEADS and len(segs) > 1 and segs[1].split() \
                and _norm_word(segs[1].split()[0]) == head_word:
            pass 
        else:
            continue
        matches.append(food)
    if not matches:
        print(f"[analyzer_node] USDA has no entry headed by '{head_word}' for query '{query}' "
              f"- falling through to Open Food Facts / AI estimate.")
        return []

    def _qmatch(qw: str, dw: set[str]) -> bool:
        if qw in dw:
            return True
        return qw in COOKING_WORDS and bool(dw & COOKING_WORDS)

    qualifiers_matched_something = any(
        _qmatch(qw, _desc_words(f.get("description", ""))) for f in matches for qw in qualifier_words
    )
    prefer_generic_variant = not qualifier_words or not qualifiers_matched_something
    cut_specified = any(
        qw not in COOKING_WORDS and qw not in RAW_WORDS
        and any(qw in _desc_words(f.get("description", "")) for f in matches)
        for qw in qualifier_words
    )

    def score(food: dict) -> tuple:
        description = food.get("description", "").lower()
        segments = [s.strip() for s in description.split(",")]
        dw = _desc_words(description)
        value = 0.0
        has_cooked = bool(dw & COOKING_WORDS)
        has_raw = bool(dw & RAW_WORDS)
        if state_pref == "cooked":
            if has_cooked:
                value -= 60
            if has_raw:
                value += 60
        elif state_pref == "raw":
            if has_raw:
                value -= 60
            if has_cooked:
                value += 60

        if prefer_generic_variant:
            if len(segments) > 1 and segments[1] in GENERIC_SEGMENT_WORDS:
                value -= 100
            elif state_pref is None and "raw" in dw:
                value -= 50
            value += 10 * sum(1 for kw in PROCESSING_PENALTY_KEYWORDS if _norm_word(kw) in dw)

        for qw in qualifier_words:
            if qw in dw:
                value -= 30 if qw in COOKING_WORDS else 5
            elif _qmatch(qw, dw):
                value -= 3  
        if "fried" in dw and "fried" not in qualifier_words:
            value += 20
        if not explicit_cooked and head_word in ("chicken", "turkey") and "roasted" in description:
            value -= 15
        if not cut_specified:
            for kw, bonus in DEFAULT_CUT_BONUS.get(head_word, ()):
                if kw in description:
                    value -= bonus
        if prefer_generic_variant or not cut_specified:
            original = food.get("description", "")
            brand_tokens = [t for t in re.findall(r"\b[A-Z]{3,}\b", original)
                            if t not in {"USDA", "NFS", "RTE", "NS"}]
            if brand_tokens:
                value += 30
            value += 25 * sum(1 for kw in NON_DEFAULT_KEYWORDS
                              if kw in dw and kw not in qualifier_words)
            value += 8 * sum(1 for kw in VARIETY_WORDS
                             if kw in dw and kw not in qualifier_words)
        return (value, len(description), food.get("fdcId") or 0)

    ranked = sorted(matches, key=score)
    print(f"[analyzer_node] '{query}' (state_pref={state_pref}) -> best: "
          f"'{ranked[0].get('description')}' (fdcId={ranked[0].get('fdcId')}), "
          f"{len(ranked)} candidates share the headword")
    return ranked

def _usda_search(term: str) -> list[dict]:
    params = {
        "api_key": USDA_API_KEY,
        "query": term,
        "pageSize": 100,
        "dataType": ["Foundation", "SR Legacy"],
    }
    resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("foods", [])


def _lookup_usda(name: str) -> list[dict]:
    """Search USDA and return ranked candidates for this food (best first).

    Queries (run in parallel, so latency = 1 request instead of 3):
      - the primary term (\"rice\"), singular and plural, because USDA's search
        doesn't treat them as equivalent;
      - primary + qualifier (\"chicken grilled\") so USDA's own relevance helps;
      - primary + \"cooked\" for foods people eat cooked, so the cooked entries
        are guaranteed to be in the candidate pool.
    Raises USDALookupError on network/HTTP failure (triggers a graph retry)."""
    primary_term = name.split(",")[0].strip()
    qualifier_segment = name.split(",", 1)[1].strip() if "," in name else ""
    plural_term = primary_term if primary_term.endswith("s") else primary_term + "s"

    terms = [primary_term]
    if plural_term != primary_term:
        terms.append(plural_term)
    if qualifier_segment:
        terms.append(f"{primary_term} {qualifier_segment}")
    head = _norm_word(primary_term.split()[0]) if primary_term.split() else ""
    if head in COOKED_BY_DEFAULT and "cooked" not in qualifier_segment:
        terms.append(f"{primary_term} cooked")

    try:
        with ThreadPoolExecutor(max_workers=len(terms)) as pool:
            pools = list(pool.map(_usda_search, terms))
    except requests.RequestException as exc:
        raise USDALookupError(f"USDA search failed for '{name}': {exc}") from exc

    seen, candidates = set(), []
    for foods in pools:
        for food in foods:
            fdc_id = food.get("fdcId")
            if fdc_id not in seen:
                seen.add(fdc_id)
                candidates.append(food)

    return _rank_usda_matches(primary_term, candidates, qualifier_hint=name) if candidates else []


def _fetch_usda_portions(fdc_id: int) -> list[dict]:
    """Non-critical accuracy enhancement — silently falls back if it fails."""
    try:
        resp = requests.get(
            USDA_DETAIL_URL.format(fdc_id=fdc_id),
            params={"api_key": USDA_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("foodPortions", [])
    except requests.RequestException:
        return []


def _extract_usda_nutrients_per_100g(food_record: dict) -> dict:
    """Per-100g nutrients from a USDA record. IMPORTANT: a nutrient that is
    genuinely absent from the record is left as None here, NOT defaulted to
    0 — that distinction is what lets _has_valid_calories() tell "USDA never
    reported this value" (bad data, should cascade) apart from "USDA
    reported an actual value of zero" (a real result, e.g. water). Missing
    non-calorie macros are normalized to 0 later, once a source has been
    chosen, since they don't gate the cascade decision the way calories do.

    Calorie fallback: USDA's newer "Foundation Foods" data type sometimes
    omits the plain "Energy" nutrient (id 1008) — the one SR Legacy entries
    reliably use — and reports only Atwater-factor variants instead:
    "Energy (Atwater General Factors)" (id 2047) or "Energy (Atwater
    Specific Factors)" (id 2048). Without this fallback, a perfectly good
    USDA match (e.g. "Apples, fuji, with skin, raw") looks like it has no
    calorie data at all and gets wrongly discarded to cascade down to OFF
    or the AI-estimate tier, even though USDA actually had the answer."""
    nutrient_map = {n.get("nutrientId"): n.get("value") for n in food_record.get("foodNutrients", [])}
    result = {key: nutrient_map.get(nid) for key, nid in NUTRIENT_IDS.items()}
    if result["calories"] is None:
        for energy_fallback_id in (2047, 2048):
            fallback_value = nutrient_map.get(energy_fallback_id)
            if fallback_value is not None:
                result["calories"] = fallback_value
                break
    return result


# Open Food Facts (fallback source — free, keyless)
def _lookup_off(name: str) -> dict | None:
    """Search Open Food Facts. Returns per-100g nutrients dict, or None.
    Non-critical: any failure here just means the item ends up unmatched,
    it doesn't abort/retry the whole pipeline (USDA is the source that
    triggers retries; OFF is a best-effort second chance)."""
    params = {
        "search_terms": name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1,
        "sort_by": "unique_scans_n",  
    }
    try:
        resp = requests.get(OFF_SEARCH_URL, params=params, headers=OFF_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[analyzer_node] OFF lookup failed for '{name}': {exc}")
        return None

    products = data.get("products", [])
    if not products:
        print(f"[analyzer_node] OFF has no products for '{name}'")
        return None

    nutriments = products[0].get("nutriments", {})
    result = {key: nutriments.get(field) for key, field in OFF_NUTRIENT_FIELDS.items()}

    # OFF entries are community-submitted and sometimes missing fields —
    # require at least calories to be present, or treat as no match.
    if result.get("calories") is None:
        print(f"[analyzer_node] OFF matched '{name}' -> '{products[0].get('product_name')}' "
              f"but it has no calorie data — treating as no match.")
        return None

    return {key: (value or 0) for key, value in result.items()}

def _has_valid_calories(per_100g: dict | None) -> bool:
    """True only if this source actually REPORTED a calorie value — zero is
    a perfectly valid, real answer (Pepsi Zero, water, black coffee); it's
    an ABSENT value (None) that means the source didn't really know and
    should not be trusted, so we cascade to the next tier instead."""
    return per_100g is not None and per_100g.get("calories") is not None


def _normalize_per_100g(per_100g: dict) -> dict:
    """Once a source has been accepted as valid (calories present), fill in
    any other individually-missing macro fields as 0 for arithmetic — those
    don't gate the cascade decision, calories alone does."""
    return {key: (value if value is not None else 0) for key, value in per_100g.items()}

def _is_plausible(per_100g: dict | None) -> bool:
    """Sanity check on a source's per-100g numbers, using the Atwater
    relationship (kcal ~ 4*protein + 4*carbs + 9*fat). This is what catches
    'rice = 0 kcal' (calories 0/missing while the macros say ~125) and
    records whose energy field belongs to a different portion/food.
    Real records land at 0.8-1.1x; fibre-heavy foods dip to ~0.8x, so the
    accepted band (0.6x - 1.35x) is deliberately loose - it only rejects
    clearly broken data."""
    if not per_100g:
        return False
    cal = per_100g.get("calories")
    if cal is None or cal < 0 or cal > 950:  # nothing has more than ~900 kcal/100g
        return False
    p, c, f = (per_100g.get(k) for k in ("protein_g", "carbs_g", "fat_g"))
    if p is None or c is None or f is None:
        return cal > 0  # macros unknown: can't cross-check; just refuse a bare 0
    atwater = 4 * p + 4 * c + 9 * f
    if atwater < 5:
        return cal < 15  # water, black coffee, diet soda...
    return 0.6 * atwater <= cal <= 1.35 * atwater

class AINutritionEstimate(BaseModel):
    calories: float = Field(description="Estimated calories per 100g of this food")
    protein_g: float = Field(description="Estimated protein grams per 100g")
    carbs_g: float = Field(description="Estimated carbohydrate grams per 100g")
    fat_g: float = Field(description="Estimated fat grams per 100g")
    sugar_g: float = Field(description="Estimated sugar grams per 100g")


AI_ESTIMATE_SYSTEM_PROMPT = """You are a nutrition estimation assistant. Given a food name, estimate \
its nutrition content PER 100 GRAMS using general nutrition knowledge.

This is a last-resort fallback used only when neither the USDA nor Open Food Facts databases returned \
a usable match for this food, so give your best realistic real-world estimate for a typical preparation \
of this food. Only return 0 for calories if the food is genuinely calorie-free (e.g. water, black \
coffee, diet soda) — never return 0 as a placeholder for "unknown"."""


def _ai_estimate_nutrition(name: str) -> dict | None:
    """Non-critical: if this fails, the item just ends up unmatched, same as
    if OFF had failed — it doesn't abort/retry the whole pipeline."""
    try:
        structured_llm = llm.with_structured_output(AINutritionEstimate)
        messages = [
            ("system", AI_ESTIMATE_SYSTEM_PROMPT),
            ("human", f"Food: {name}"),
        ]
        result: AINutritionEstimate = structured_llm.invoke(messages)
        return result.model_dump()
    except Exception as exc:
        print(f"[analyzer_node] AI nutrition estimate failed for '{name}': {exc}")
        return None
    
# Unit conversion (shared by both sources)

def _convert_to_grams(item: dict, usda_portions: list[dict]) -> tuple[float, bool]:
    """Convert an item's quantity+unit to grams. Returns (grams, was_estimated).

    Preference order:
      1. unit is already "g" or "kg" -> exact, this is the user's own
         stated weight and always wins.
      2. unit is "piece" and the food is a known countable produce item
         (apple, banana, egg, ...) -> a fixed reference weight, exact and
         deterministic (not an LLM guess), per the standard reference
         sizes used for that produce's own per-100g nutrient data.
      3. USDA's own portion data for this specific food -> exact (only
         applies to USDA-sourced items; OFF doesn't provide this).
      4. The parser's LLM-estimated grams for this specific food+quantity
         -> approximate, but food-aware.
      5. Generic per-unit table -> last resort.
    """
    unit = item["unit"].lower()
    quantity = item["quantity"]

    if unit == "g":
        return quantity, False
    if unit == "kg":
        return quantity * 1000, False

    if unit == "piece":
        count_based_grams = _get_count_based_grams(item["name"])
        if count_based_grams is not None:
            return quantity * count_based_grams, False

    for portion in usda_portions:
        measure_name = portion.get("measureUnit", {}).get("name", "").lower()
        if unit in measure_name or measure_name in unit:
            amount = portion.get("amount") or 1
            gram_weight = portion.get("gramWeight") or 0
            if amount and gram_weight:
                return (quantity / amount) * gram_weight, False

    estimated_grams = item.get("estimated_grams")
    if estimated_grams:
        return estimated_grams, True

    grams_per_unit = GENERIC_GRAMS_PER_UNIT.get(unit, GENERIC_GRAMS_PER_UNIT["piece"])
    return quantity * grams_per_unit, True



def _build_item(name: str, per_100g: dict, item: dict, portions: list[dict], source: str,
                matched_to: str | None, alternatives: list[dict] | None = None) -> dict:
    per_100g = _normalize_per_100g(per_100g)
    grams, was_estimated = _convert_to_grams(item, portions)
    scale = grams / 100
    return {
        "name": name,
        "matched": True,
        "needs_confirmation": False,
        "unit_conversion_estimated": was_estimated,
        "source": source,
        "matched_to": matched_to,         
        "grams": round(grams, 1),
        "alternatives": alternatives or [],
        "calories": per_100g["calories"] * scale,
        "protein_g": per_100g["protein_g"] * scale,
        "carbs_g": per_100g["carbs_g"] * scale,
        "fat_g": per_100g["fat_g"] * scale,
        "sugar_g": per_100g["sugar_g"] * scale,
    }


def analyze_nutrition(state: FoodState) -> FoodState:
    state["retry_count"] = state.get("retry_count", 0) + 1

    items_result = []

    try:
        for item in state["parser_output"]:
            name = item["name"]

            cached = get_cached_match(name)
            if cached is not None and _is_plausible(cached.get("per_100g")):
                print(f"[analyzer_node] '{name}' from CACHE -> '{cached['description']}' "
                      f"({cached['per_100g'].get('calories')} kcal/100g)")
                items_result.append(_build_item(name, cached["per_100g"], item, [], "cached",
                                                cached.get("description")))
                continue

            ranked = _lookup_usda(name)
            chosen, per_100g = None, None
            for food in ranked[:8]:
                candidate = _extract_usda_nutrients_per_100g(food)
                if _has_valid_calories(candidate) and _is_plausible(candidate):
                    chosen, per_100g = food, candidate
                    break
                print(f"[analyzer_node] skipping USDA '{food.get('description')}' for '{name}': "
                      f"implausible data {candidate}")

            if chosen is not None:
                fdc_id = chosen.get("fdcId")
                portions = _fetch_usda_portions(fdc_id) if fdc_id else []
                alts = []
                for food in ranked[:4]:
                    if food is chosen:
                        continue
                    alt = _extract_usda_nutrients_per_100g(food)
                    if alt.get("calories") is None or food.get("description") == chosen.get("description"):
                        continue  
                    alts.append({"description": food.get("description"),
                                 "calories_per_100g": alt.get("calories")})
                normalized = _normalize_per_100g(per_100g)
                save_confirmed_match(name, fdc_id, chosen.get("description", ""), normalized)
                print(f"[analyzer_node] '{name}' final: USDA '{chosen.get('description')}' "
                      f"({normalized['calories']} kcal/100g)")
                items_result.append(_build_item(name, per_100g, item, portions, "usda",
                                                chosen.get("description"), alts[:3]))
                continue

            off_nutrients = _lookup_off(name)
            if _has_valid_calories(off_nutrients) and _is_plausible(off_nutrients):
                items_result.append(_build_item(name, off_nutrients, item, [], "off", None))
                continue

            ai_nutrients = _ai_estimate_nutrition(name)
            if _has_valid_calories(ai_nutrients) and _is_plausible(ai_nutrients):
                items_result.append(_build_item(name, ai_nutrients, item, [], "ai_estimate", None))
                continue

            items_result.append({
                "name": name,
                "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0,
                "matched": False,
                "needs_confirmation": False,
                "unit_conversion_estimated": False,
            })
    except USDALookupError as exc:
        print(f"[analyzer_node] {exc}")
        state["error"] = str(exc)
        return state

    totals = {
        "calories": sum(r["calories"] for r in items_result),
        "protein_g": sum(r["protein_g"] for r in items_result),
        "carbs_g": sum(r["carbs_g"] for r in items_result),
        "fat_g": sum(r["fat_g"] for r in items_result),
        "sugar_g": sum(r["sugar_g"] for r in items_result),
    }

    protein_kcal = totals["protein_g"] * 4
    carbs_kcal = totals["carbs_g"] * 4
    fat_kcal = totals["fat_g"] * 9
    macro_kcal_total = protein_kcal + carbs_kcal + fat_kcal or 1

    macro_percentages = {
        "protein_pct": round(protein_kcal / macro_kcal_total * 100, 1),
        "carbs_pct": round(carbs_kcal / macro_kcal_total * 100, 1),
        "fat_pct": round(fat_kcal / macro_kcal_total * 100, 1),
    }

    state["error"] = None
    state["analyzer_output"] = {
        "items": items_result,
        "totals": totals,
        "macro_percentages": macro_percentages,
    }
    return state
