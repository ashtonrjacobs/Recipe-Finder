from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
def load_dataset(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

# Preprocess dataset
def preprocess_ingredients(dataset):
    dataset['ingredients'] = dataset['ingredients'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    return dataset

# Find recipes
def find_recipes(ingredients, dataset, count_matrix, vectorizer):
    ingredients = [ingredients]
    user_ingredients_vector = vectorizer.transform(ingredients)
    cosine_similarities = cosine_similarity(user_ingredients_vector, count_matrix).flatten()
    similar_recipes_indices = cosine_similarities.argsort()[-5:][::-1]
    return dataset.iloc[similar_recipes_indices], cosine_similarities[similar_recipes_indices]

# Initialize dataset and CountVectorizer
dataset = load_dataset('recipes.csv')
if dataset is not None:
    dataset = preprocess_ingredients(dataset)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dataset['ingredients'])

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        ingredients = request.form.get('ingredients', '').strip().lower()
        if dataset is not None:
            matching_recipes, similarities = find_recipes(ingredients, dataset, count_matrix, vectorizer)
            results = []
            for i, (index, recipe) in enumerate(matching_recipes.iterrows()):
                results.append({
                    "recipe_name": recipe['recipe_name'],
                    "ingredients": recipe['ingredients'],
                    "similarity": round(similarities[i], 2)
                })
    return render_template('index.html', results=results)



if __name__ == '__main__':
    app.run(debug=True)