import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (assuming it has 'name' and 'ingredients' columns)
def load_dataset(filepath):
    """
    Load the recipe dataset from a CSV file.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def preprocess_ingredients(dataset):
    """
    Combine all ingredients into a string for each recipe (if not already formatted this way).
    Assumes the 'ingredients' column is a list of ingredients.
    """
    # Assuming ingredients are a list of strings, we join them into one string per recipe
    dataset['ingredients'] = dataset['ingredients'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    return dataset

def find_recipes(ingredients, dataset, count_matrix, vectorizer):
    """
    Use cosine similarity to find recipes that match the given ingredients using CountVectorizer.
    """
    # Convert user input ingredients to lowercase and create a Count vector for it
    ingredients = [ingredients]
    user_ingredients_vector = vectorizer.transform(ingredients)
    
    # Calculate cosine similarity between user ingredients and all recipes
    cosine_similarities = cosine_similarity(user_ingredients_vector, count_matrix).flatten()
    
    # Get indices of the most similar recipes
    similar_recipes_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 results
    
    # Retrieve and return the matching recipes
    return dataset.iloc[similar_recipes_indices], cosine_similarities[similar_recipes_indices]

def display_recipes(recipes, similarities):
    """
    Display the recipes and their similarity scores in the console.
    """
    if not recipes.empty:
        print(f"\nFound {len(recipes)} recipe(s) with similar ingredients:")
        for i, (index, recipe) in enumerate(recipes.iterrows(), start=1):
            print(f"\nRecipe {i}: {recipe['recipe_name']}")
            print(f"Ingredients: {recipe['ingredients']}")
            print(f"Similarity Score: {similarities[i-1]:.2f}")
    else:
        print("\nNo recipes found with the given ingredients.")

def main():
    """
    Main function to run the console app.
    """
    # Load dataset (adjust the file path as needed)
    dataset = load_dataset('recipes.csv')
    
    if dataset is not None:
        # Preprocess ingredients for vectorization
        dataset = preprocess_ingredients(dataset)
        
        # Initialize the Count Vectorizer
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(dataset['ingredients'])
        
        # Get user input
        print("Welcome to the Ingredient-Based Recipe App!")
        ingredients = input("Enter ingredients (comma-separated): ").strip().lower()
        
        # Find matching recipes
        matching_recipes, similarities = find_recipes(ingredients, dataset, count_matrix, vectorizer)
        
        # Display the matching recipes
        display_recipes(matching_recipes, similarities)

if __name__ == "__main__":
    main()
