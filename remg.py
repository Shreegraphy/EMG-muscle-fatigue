import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load a single CSV file instead of all files
file_path = r'D:\Research\EMG\emg\Fatigue Data\Fatigue Data\Study2\median\System 1\U20Ex3.csv'  # Change to your specific file
data = pd.read_csv(file_path, header=None, names=['Time', 'EMG', 'Fatigue'])

# Check for missing values and balance
print(data.isnull().sum())
print(data['Fatigue'].value_counts())

# Step 2: Split into features (X) and labels (y)
X = data[['Time', 'EMG']].values  # Features
y = data['Fatigue'].values  # Labels

# Step 3: Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for RNN (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Step 5: Build the RNN model with improvements
def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),  # Use LSTM instead of SimpleRNN
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 6: Create the model
input_shape = (X_train.shape[1], X_train.shape[2])
rnn_model = build_rnn_model(input_shape)

# Step 7: Train the model with increased epochs and batch size
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = rnn_model.fit(X_train, y_train, epochs=100, batch_size=16, 
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

# Step 8: Evaluate the model
test_loss, test_acc = rnn_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Step 9: Make predictions
predictions = rnn_model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Step 10: Compute confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)
print(f"Confusion Matrix:\n{conf_matrix}")

# Step 11: Classification report for more metrics
print(classification_report(y_test, predicted_labels))

# Step 12: Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Non-Fatigue', 'Fatigue'], yticklabels=['Non-Fatigue', 'Fatigue'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for RNN Model')
plt.show()

# Step 13: Predict fatigue based on user input
def predict_fatigue(model):
    while True:
        try:
            # Get user input
            time_input = float(input("Enter Time: "))
            emg_input = float(input("Enter EMG: "))
            
            # Create new data point
            new_sample = np.array([[time_input, emg_input]])
            
            # Normalize the new data using the same scaler
            new_sample = scaler.transform(new_sample)
            
            # Reshape the new data for RNN (samples, time steps, features)
            new_sample = new_sample.reshape((1, 1, new_sample.shape[1]))
            
            # Make a prediction
            prediction = model.predict(new_sample)
            predicted_class = (prediction > 0.5).astype(int)  # Convert probability to binary class
            
            # Print the prediction
            print(f'Predicted class for Time: {time_input}, EMG: {emg_input} (0 for Non-Fatigue, 1 for Fatigue): {predicted_class[0][0]}')
        
        except ValueError:
            print("Invalid input. Please enter numeric values for Time and EMG.")
        
        cont = input("Do you want to predict another sample? (yes/no): ")
        if cont.lower() != 'yes':
            break

# Call the function to predict based on user input
predict_fatigue(rnn_model)
