document.getElementById('search-button').addEventListener('click', () => {
    const ingredients = document.getElementById('ingredients-input').value;
    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ingredients })
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';
        if (data.length > 0) {
            data.forEach(recipe => {
                const recipeDiv = document.createElement('div');
                recipeDiv.classList.add('recipe');
                recipeDiv.innerHTML = `
                    <h2>${recipe.recipe_name}</h2>
                    <p>Ingredients: ${recipe.ingredients}</p>
                    <p>Similarity: ${recipe.similarity}</p>
                `;
                resultsDiv.appendChild(recipeDiv);
            });
        } else {
            resultsDiv.innerHTML = '<p>No matching recipes found.</p>';
        }
    });
});
