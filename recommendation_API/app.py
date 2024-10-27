from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model
model = joblib.load('plant_recommendation_model.joblib')


def preprocess_input(user_input):
    """Preprocess the user input for prediction."""
    input_df = pd.DataFrame([user_input])
    return model.named_steps['preprocessor'].transform(input_df)


def recommend_based_on_similarity(input_transformed):
    """Recommend plants based on cosine similarity for unknown inputs."""
    all_plants = model.named_steps['classifier'].classes_
    all_plants_df = pd.DataFrame(all_plants, columns=['Plant Name'])
    all_plants_transformed = model.named_steps['preprocessor'].transform(all_plants_df)
    similarities = cosine_similarity(input_transformed, all_plants_transformed)
    similar_indices = np.argsort(similarities[0])[::-1][:5]
    return all_plants[similar_indices].tolist()


def recommend_based_on_probabilities(input_df):
    """Recommend plants based on predicted probabilities."""
    predicted_probs = model.predict_proba(input_df)
    plant_indices = np.argsort(predicted_probs[0])[::-1][:5]
    return model.named_steps['classifier'].classes_[plant_indices].tolist()


@app.route('/recommend', methods=['POST'])
def recommend_plants():
    user_input = request.json

    # Validate input
    if not user_input:
        return jsonify({"error": "Invalid input"}), 400

    # Preprocess user input
    try:
        input_transformed = preprocess_input(user_input)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Check for unknowns in input and recommend accordingly
    if 'Unknown' in user_input.values():
        recommended_plants = recommend_based_on_similarity(input_transformed)
    else:
        input_df = pd.DataFrame([user_input])
        recommended_plants = recommend_based_on_probabilities(input_df)

    return jsonify(recommended_plants)


if __name__ == '__main__':
    app.run(debug=True)
