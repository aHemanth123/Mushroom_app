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

# Feature explanations (keys must match feature names exactly)
feature_explanations = {
    "bruises": "Does the mushroom bruise when touched?",
    "cap-color": "Color of the mushroom cap.",
    "cap-shape": "Shape of the mushroom cap (e.g., bell, convex).",
    "cap-surface": "Surface texture of the cap.",
    "gill-attachment": "How the gills are attached to the stalk.",
    "gill-color": "Color of the gills.",
    "gill-size": "Size of the gills.",
    "gill-spacing": "Distance between gills.",
    "habitat": "The environment where the mushroom grows.",
    "odor": "Odor emitted by the mushroom.",
    "population": "Estimated local mushroom population.",
    "ring-number": "Number of rings on the stalk.",
    "ring-type": "Type/form of the ring.",
    "spore-print-color": "Color left by spore print.",
    "stalk-color-above-ring": "Color of the stalk above the ring.",
    "stalk-color-below-ring": "Color of the stalk below the ring.",
    "stalk-root": "Type of base/root of the stalk.",
    "stalk-shape": "Shape of the stalk (enlarging or tapering).",
    "stalk-surface-above-ring": "Surface texture of the stalk above the ring.",
    "stalk-surface-below-ring": "Surface texture of the stalk below the ring.",
    "veil-color": "Color of the veil covering the gills.",
    "veil-type": "Type of veil (e.g., partial)."
}

# Label mappings
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
    "veil-type": {"p": "Partial"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"},
    "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"},
}

# App UI
st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄")
st.title("🍄 Mushroom Edibility Prediction")
st.markdown("###  🧾 Mushroom Features")

user_input = {}
all_selected = True
missing_features = []

sorted_features = sorted(set(col.split("_")[0] for col in feature_cols))

for i, feature in enumerate(sorted_features, start=1):
    clean_name = feature.replace("-", " ").capitalize()
    numbered_name = f"{i}. {clean_name}"
    explanation = feature_explanations.get(feature, "No description available.")
    heading = f"**{numbered_name}** – *{explanation}*"

    if feature in label_mappings:
        options = label_mappings[feature]
        display_options = ["Select..."] + list(options.values())
        selected_full = st.selectbox(heading, display_options, index=0)

        if selected_full == "Select...":
            all_selected = False
            missing_features.append(numbered_name)
        else:
            reverse_mapping = {v: k for k, v in options.items()}
            user_input[feature] = reverse_mapping[selected_full]
    else:
        categories = sorted({col.split("_")[1] for col in feature_cols if col.startswith(feature)})
        display_options = ["Select..."] + categories
        selected = st.selectbox(heading, display_options, index=0)

        if selected == "Select...":
            all_selected = False
            missing_features.append(numbered_name)
        else:
            user_input[feature] = selected

# Prediction button
if st.button("Predict"):
    if not all_selected:
        st.warning("⚠️ Please select all required features before predicting.")
        st.error("Missing selections for:\n\n" + "\n".join(missing_features))
    else:
        input_df = pd.DataFrame(columns=feature_cols)
        input_df.loc[0] = [0] * len(feature_cols)
        for col in feature_cols:
            f, val = col.split("_")
            if user_input[f] == val:
                input_df.at[0, col] = 1

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.markdown("###  🧠 Prediction Result")
        if prediction == 1:
            st.error("☠️ The mushroom is **Poisonous**!")
        else:
            st.success("🍄 The mushroom is **Edible**!")
