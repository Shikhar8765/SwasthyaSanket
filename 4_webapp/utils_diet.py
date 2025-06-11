import pandas as pd

def generate_diet_chart(age, bmi, smokes, drinks, history):
    diet_plan = []

    if bmi > 30:
        diet_plan += ["Whole grains", "Leafy greens", "Low-fat dairy", "Avoid sugary foods"]
    elif bmi < 18.5:
        diet_plan += ["High-calorie healthy foods", "Nuts & seeds", "Milk products"]
    else:
        diet_plan += ["Balanced diet", "Vegetables", "Fruits", "Whole grains"]

    if drinks:
        diet_plan += ["Reduce alcohol", "Hydration with water & herbal teas"]

    if smokes:
        diet_plan += ["Antioxidant-rich foods", "Citrus fruits", "Green tea"]

    if history:
        diet_plan += ["Limit processed food", "Low-salt intake", "Lean protein sources"]

    if age > 60:
        diet_plan += ["Calcium-rich foods", "Vitamin D", "Soft fiber-rich meals"]

    # Remove duplicates and keep order
    diet_plan = list(dict.fromkeys(diet_plan))

    return pd.DataFrame({"Recommended Food Items": diet_plan})
