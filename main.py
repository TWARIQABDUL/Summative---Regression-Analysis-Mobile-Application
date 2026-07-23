import numpy as np
import pandas as pd
# df  = pd.read_csv("/dataset/used_phone_price_prediction_1M.csv")
df = pd.read_csv("dataset/used_phone_price_prediction_1M.csv")

print(df.head())
print(df.columns)
print(df.info())

# def gradient_decent(x, y, w, b, rate=0.0001, epochs=10000000):
#     # Convert standard Python lists into NumPy arrays for lightning-fast math
#     X = np.array(x)
#     Y = np.array(y)
#     W = np.array(w)
#     m = len(X)

#     for e in range(epochs):
#         # 1. Calculate guesses for ALL students simultaneously!
#         # np.dot(X, W) does the (w0*x0 + w1*x1) for every row automatically.
#         guesses = np.dot(X, W) + b
        
#         # 2. Calculate errors for ALL students simultaneously
#         errors = guesses - Y

#         # 3. Calculate gradients for ALL weights simultaneously
#         # X.T (Transpose) flips the table to perfectly align the features with the errors
#         grad_W = np.dot(X.T, errors) / m
#         grad_b = np.sum(errors) / m

#         # 4. Update the weights and bias
#         W -= rate * grad_W
#         b -= rate * grad_b

#     # Return the final calculated arrays
#     return W, b

# # [Hours Studied, Attendance %]
# x_train = [
#   [12, 90],  # Student 0
#   [8,  75],  # Student 1
#   [15, 95],  # Student 2
#   [5,  60]   # Student 3
# ]

# # The actual marks stay as a simple 1D list (one answer per student)
# y_train = [85, 70, 96, 50]

# # 1. Convert to NumPy arrays
# X = np.array(x_train)
# Y = np.array(y_train)

# # 2. FEATURE SCALING (Z-Score Normalization)
# # Find the mean and standard deviation for each column (Hours, Attendance)
# x_mean = np.mean(X, axis=0) 
# x_std = np.std(X, axis=0)

# # Scale the training data
# X_scaled = (X - x_mean) / x_std

# weight = [0.0, 0.0]
# bias = 0

# # 3. Train using the SCALED data! 
# # Because the data is scaled, we can safely turn the learning rate WAY up to 0.1
# final_w, final_b = gradient_decent(X_scaled, Y, weight, bias, rate=0.1, epochs=1000)

# print(f"The scaled weights are: w = {final_w}, b = {final_b:.2f}")

# # ==========================================
# # 4. PREDICTING FOR A NEW STUDENT
# # ==========================================
# # Let's say a new student studies 10 hours with 80% attendance
# x_new = np.array([15, 95])

# # THE GOLDEN RULE OF SCALING: 
# # You MUST scale the new student using the EXACT SAME mean and std from your training data!
# x_new_scaled = (x_new - x_mean) / x_std

# # Predict using the trained weights and the scaled new student
# prediction = np.dot(final_w, x_new_scaled) + final_b
# print(f"Predicted mark for {x_new[0]} hours and {x_new[1]}% attendance: {prediction:.2f}")