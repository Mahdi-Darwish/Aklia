ACTIVITY_FACTORS = {
    "sedentary": 1.2,      # desk job, little or no exercise
    "light": 1.375,        # light exercise 1-3 days/week
    "moderate": 1.55,      # moderate exercise 3-5 days/week
    "active": 1.725,       # hard exercise 6-7 days/week
    "very_active": 1.9,    # very hard training / physical job
}
ACTIVITY_LABELS = {
    "sedentary": "sedentary", "light": "lightly active", "moderate": "moderately active",
    "active": "very active", "very_active": "extra active",
}
ACTIVITY_ALIASES = {
    "lazy": "sedentary", "inactive": "sedentary", "low": "sedentary",
    "lightly active": "light", "light active": "light",
    "moderately active": "moderate", "medium": "moderate",
    "very active": "active", "active": "active", "high": "active",
    "super active": "very_active", "extra active": "very_active", "very_active": "very_active",
}

LOSE_WORDS = ("lose", "loose", "losing", "fat", "cut", "cutting", "slim", "weight loss",
              "deficit", "diet", "shred", "نزول", "خسارة", "تنشيف", "حرق", "دهون", "تخسيس", "نحافة")
GAIN_WORDS = ("gain", "muscle", "bulk", "bulking", "mass", "surplus", "build", "strength",
              "زيادة", "تضخيم", "عضل", "كتلة", "بناء")
MAINTAIN_WORDS = ("maintain", "maintenance", "keep", "healthy", "stay", "توازن", "ثبات", "حافظ")


def classify_goal(goal: str) -> str:
    """'lose' | 'gain' | 'maintain' | 'unspecified' from free text."""
    g = (goal or "").lower()
    if not g.strip():
        return "unspecified"
    lose = any(w in g for w in LOSE_WORDS)
    gain = any(w in g for w in GAIN_WORDS)
    if lose and gain:
        return "maintain"
    if lose:
        return "lose"
    if gain:
        return "gain"
    if any(w in g for w in MAINTAIN_WORDS):
        return "maintain"
    return "unspecified"


def normalize_profile(raw: dict | None) -> dict | None:
    """Returns a clean profile or None if anything required is missing/out of range."""
    if not raw:
        return None
    try:
        age = int(float(raw.get("age")))
        height_cm = float(raw.get("height_cm"))
        weight_kg = float(raw.get("weight_kg"))
    except (TypeError, ValueError):
        return None
    gender = str(raw.get("gender") or "").strip().lower()
    if gender in ("m", "man", "boy"):
        gender = "male"
    if gender in ("f", "woman", "girl"):
        gender = "female"
    activity = str(raw.get("activity") or "").strip().lower()
    activity = ACTIVITY_ALIASES.get(activity, activity)
    if gender not in ("male", "female") or activity not in ACTIVITY_FACTORS:
        return None
    if not (14 <= age <= 100 and 120 <= height_cm <= 230 and 30 <= weight_kg <= 300):
        return None
    return {"age": age, "gender": gender, "height_cm": height_cm,
            "weight_kg": weight_kg, "activity": activity}


def compute_targets(profile: dict | None, goal: str) -> dict | None:
    profile = normalize_profile(profile)
    if profile is None:
        return None

    w, h, a = profile["weight_kg"], profile["height_cm"], profile["age"]
    male = profile["gender"] == "male"

    bmr = 10 * w + 6.25 * h - 5 * a + (5 if male else -161)          # Mifflin-St Jeor
    tdee = bmr * ACTIVITY_FACTORS[profile["activity"]]
    bmi = w / ((h / 100) ** 2)

    goal_type = classify_goal(goal)
    notes = []

    if goal_type == "lose" and bmi < 18.5:
        goal_type_used = "maintain"
        notes.append("BMI is below 18.5, so no calorie deficit is suggested.")
    else:
        goal_type_used = goal_type

    if goal_type_used == "lose":
        calories = max(tdee - 500, tdee * 0.80)                       # never more than a 20% deficit
        floor = 1500 if male else 1200
        if calories < floor:
            calories = min(floor, tdee)
            notes.append(f"Target raised to a safe minimum of {floor} kcal.")
        protein_per_kg = 2.0
    elif goal_type_used == "gain":
        calories = tdee + 300
        protein_per_kg = 1.8
    else:                                                              # maintain / unspecified
        calories = tdee
        protein_per_kg = 1.6
        if goal_type == "unspecified":
            notes.append("No clear goal detected, so targets are set at maintenance.")

    protein_g = protein_per_kg * w
    fat_g = 0.25 * calories / 9
    carbs_g = max(0.0, (calories - 4 * protein_g - 9 * fat_g) / 4)

    return {
        "goal_type": goal_type_used,
        "activity_label": ACTIVITY_LABELS[profile["activity"]],
        "bmi": round(bmi, 1),
        "bmr": round(bmr),
        "maintenance_kcal": round(tdee),
        "target_kcal": round(calories),
        "protein_g": round(protein_g),
        "carbs_g": round(carbs_g),
        "fat_g": round(fat_g),
        "notes": notes,
    }