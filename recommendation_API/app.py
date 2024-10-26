from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model
model = joblib.load('plant_recommendation_model.joblib')


@app.route('/recommend', methods=['POST'])
def recommend_plants():
    user_input = request.json
    input_df = pd.DataFrame([user_input])

    # Preprocess user input
    input_transformed = model.named_steps['preprocessor'].transform(input_df)

    # Check for unknowns
    if 'Unknown' in input_df.values:
        all_plants_transformed = model.named_steps['preprocessor'].transform(model.named_steps['classifier'].classes_)
        similarities = cosine_similarity(input_transformed, all_plants_transformed)
        similar_indices = np.argsort(similarities[0])[::-1][:5]
        recommended_plants = model.named_steps['classifier'].classes_[similar_indices].tolist()
    else:
        predicted_probs = model.predict_proba(input_df)
        plant_indices = np.argsort(predicted_probs[0])[::-1][:5]
        recommended_plants = model.named_steps['classifier'].classes_[plant_indices].tolist()

    return jsonify(recommended_plants)


if __name__ == '__main__':
    app.run(debug=True)
