import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model artifacts
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Define mapping from encoded values to full names
label_mappings = {
    "cap-shape": {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"},
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"},
    "cap-color": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "bruises": {"t": "Bruises", "f": "No"},
    "odor": {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"},
    "gill-attachment": {"a": "Attached", "f": "Free"},
    "gill-spacing": {"c": "Close", "w": "Crowded"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted", "?": "Missing"},
    "stalk-surface-above-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-color-above-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-color-below-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"},
    "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"},
}

# Streamlit App
st.title("🍄 Mushroom Edibility Prediction")
st.markdown("#### Provide Mushroom Features:")

# Collect user input with full-form labels
user_input = {}
for feature in sorted(set(col.split("_")[0] for col in feature_cols)):
    if feature in label_mappings:
        options = label_mappings[feature]
        reverse_mapping = {v: k for k, v in options.items()}
        selected_full = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", options.values())
        user_input[feature] = reverse_mapping[selected_full]
    else:
        categories = sorted({col.split("_")[1] for col in feature_cols if col.startswith(feature)})
        selected = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", categories)
        user_input[feature] = selected

# Prediction only when button is clicked
if st.button("Predict"):
    # One-hot encode
    input_df = pd.DataFrame(columns=feature_cols)
    input_df.loc[0] = [0] * len(feature_cols)
    for col in feature_cols:
        f, val = col.split("_")
        if user_input[f] == val:
            input_df.at[0, col] = 1

    # Scale and Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Display result
    st.markdown("### 🧠 Prediction Result:")
    if prediction == 1:
        st.error("☠️ The mushroom is **Poisonous**!")
    else:
        st.success("🍄 The mushroom is **Edible**!")
