import json

with open('linear_regression.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Add RandomForest and DecisionTree to imports
        if any('from sklearn.linear_model import' in line for line in cell['source']):
            cell['source'].append('\nfrom sklearn.ensemble import RandomForestRegressor\n')
            cell['source'].append('from sklearn.tree import DecisionTreeRegressor\n')
        
        # Add RandomForest and DecisionTree to models dict
        if any('models = {' in line for line in cell['source']):
            for i, line in enumerate(cell['source']):
                if '"Lasso Regression": Lasso(alpha=0.01, random_state=42)' in line:
                    cell['source'][i] = line.rstrip('\n') + ',\n'
                    cell['source'].insert(i+1, '    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),\n')
                    cell['source'].insert(i+2, '    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)\n')
                    break

# Add explanation cell for best model criteria
explanation_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Best Model Selection Criteria\n",
        "The best performing model is chosen based on the lowest **Root Mean Squared Error (RMSE)** on the test dataset. A lower RMSE indicates that the model's predictions are closer to the actual observed crop yields."
    ]
}

# Add prediction on one data point
prediction_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ---------------------------------------------------------\n",
        "# REQUIREMENT: Make a prediction on one data point from the test data\n",
        "# ---------------------------------------------------------\n",
        "single_row_scaled = X_test_scaled[0].reshape(1, -1)\n",
        "actual_yield = Y_test[0]\n",
        "predicted_yield = best_model.predict(single_row_scaled)[0]\n",
        "print(\"--- SINGLE DATA POINT PREDICTION ---\")\n",
        "print(f\"Actual Yield: {actual_yield:.4f} tons/hectare\")\n",
        "print(f\"Predicted Yield: {predicted_yield:.4f} tons/hectare\")\n",
        "print(f\"Difference: {abs(actual_yield - predicted_yield):.4f} tons/hectare\")\n"
    ]
}

nb['cells'].append(explanation_cell)
nb['cells'].append(prediction_cell)

with open('linear_regression.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
