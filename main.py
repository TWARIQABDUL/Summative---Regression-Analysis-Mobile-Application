import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading data...")
df = pd.read_csv("dataset/used_phone_price_prediction_1M.csv")

# ==========================================
# 1. FEATURE SELECTION (Dropping Noise)
# ==========================================
columns_to_drop = [
    'usage_hours_per_day', 
    'purchase_year', 
    'os_type', 
    'model' 
]
df = df.drop(columns=columns_to_drop)

# ==========================================
# 2. LABEL ENCODING (Ordered Data)
# ==========================================
condition_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3, 'Mint': 4}
df['condition'] = df['condition'].map(condition_mapping)

tier_mapping = {'Tier3': 0, 'Tier2': 1, 'Tier1': 2} 
df['city_tier'] = df['city_tier'].map(tier_mapping)

# ==========================================
# 3. ONE-HOT ENCODING (Unordered Data)
# ==========================================
df = pd.get_dummies(df, columns=['brand', 'seller_type'], drop_first=True)

# Convert all columns to float to keep the math engine happy
df = df.astype(float) 

print("\nData cleaning complete! Here is the new dataset info:")
print(df.info())

# ==========================================
# 4. SEPARATE FEATURES (X) AND TARGET (Y)
# ==========================================
# Y is what we want to predict. X is everything else.
Y = df['resale_price'].values
X = df.drop(columns=['resale_price']).values

print("\nShape of X (Features):", X.shape)
print("Shape of Y (Target):", Y.shape)

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================
# test_size=0.2 means 20% of the data goes to the test set, 80% to training.
# random_state=42 ensures we get the exact same random shuffle every time we run the script.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nData successfully split!")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples:  {len(X_test)}")

# ==========================================
# 6. FEATURE SCALING (Z-Score)
# ==========================================
print("\nScaling features...")
# Since we are already using scikit-learn, we can use its built-in scaler 
# instead of writing out the numpy mean/std math manually!
scaler = StandardScaler()

# 1. We "fit" (calculate mean & std) AND scale the Training data
X_train_scaled = scaler.fit_transform(X_train)

# 2. We ONLY scale the Testing data using the mean & std we found in the training data
# We NEVER fit() the scaler on the test data!
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. TRAIN THE MODEL (Gradient Descent)
# ==========================================
def gradient_descent(x, y, w, b, rate=0.1, epochs=1000000):
    m = len(x)
    for e in range(epochs):
        # 1. Predictions for all 800,000 rows at once!
        guesses = np.dot(x, w) + b
        
        # 2. Errors
        errors = guesses - y
        
        # 3. Gradients
        grad_w = np.dot(x.T, errors) / m
        grad_b = np.sum(errors) / m
        
        # 4. Updates
        w -= rate * grad_w
        b -= rate * grad_b
        
        # Print progress every 100 epochs to ensure it's not exploding
        if e % 100 == 0:
            # Mean Squared Error (Average of errors squared)
            mse = np.mean(errors ** 2)
            print(f"Epoch {e:4d} | Mean Squared Error: {mse:,.2f}")
            
    return w, b

# We have 28 features now, so we need 28 weights! np.zeros creates an array of 28 zeros.
initial_w = np.zeros(X_train_scaled.shape[1])
initial_b = 0.0

print("\nStarting Gradient Descent on 800,000 rows...")
final_w, final_b = gradient_descent(X_train_scaled, Y_train, initial_w, initial_b, rate=0.1, epochs=1000)

print("\nTraining Complete!")
print(f"Final Bias (Average Phone Price): {final_b:,.2f}")

# ==========================================
# 8. EVALUATE ON THE TEST SET
# ==========================================
print("\n--- TEST SET EVALUATION ---")
# 1. Make predictions on the 200,000 unseen test rows using our newly learned weights!
test_guesses = np.dot(X_test_scaled, final_w) + final_b

# 2. Calculate the errors
test_errors = test_guesses - Y_test

# 3. Calculate Mean Squared Error (MSE)
test_mse = np.mean(test_errors ** 2)

# 4. Calculate Root Mean Squared Error (RMSE)
test_rmse = np.sqrt(test_mse)

print(f"Test Mean Squared Error (MSE): {test_mse:,.2f}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse:,.2f} currency units")

# Let's peek at the first 5 predictions to see how it did!
print("\nSample Predictions vs Actual Prices:")
for i in range(5):
    predicted = test_guesses[i]
    actual = Y_test[i]
    difference = abs(predicted - actual)
    print(f"Predicted: {predicted:,.2f} | Actual: {actual:,.2f} | Off by: {difference:,.2f}")