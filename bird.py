import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load audio file (limit duration to 5 seconds)
        y, sr = librosa.load(file_path, duration=5.0)
        print(f"Loaded audio: {file_path}, duration: {len(y)} samples")  # Debugging

        # Extract MFCC (Mel Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.size == 0:
            print(f"Warning: Empty MFCC for file {file_path}")
            return None  # Return None if no features are extracted

        # Normalize features by taking the mean of the MFCCs across time
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None  # Return None if there's an error

# Function to load data from the specified directory (including nested directories)
def load_data(directory):
    features = []
    labels = []
    for root, dirs, files in os.walk(directory):  # Recursively walk through directories
        for filename in files:
            if filename.endswith('.wav'):  # Only process .wav files
                file_path = os.path.join(root, filename)
                print(f"Found file: {file_path}")  # Debugging
                feature = extract_features(file_path)
                if feature is not None:  # Ensure features are not None
                    label = os.path.basename(root)  # Use the directory name as the label
                    features.append(feature)
                    labels.append(label)
                else:
                    print(f"Warning: Feature extraction failed for {file_path}")
    features = np.array(features)
    labels = np.array(labels)
    print(f"Loaded {len(features)} features and {len(labels)} labels.")  # Debugging
    return features, labels

# Main function
if __name__ == "__main__":
    # Ask for the path to the 'bird_sounds' directory
    directory = input("Please enter the path to the 'bird_sounds' directory: ")

    # Load the dataset
    X, y = load_data(directory)

    if X.size == 0:
        print("Error: No features were loaded. Exiting.")
        exit()

    # Normalize the features
    X_normalized = X / np.max(X, axis=0)  # Normalize to [0, 1]

    # Encode the labels (bird species)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

    # Build a simple neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer with softmax for classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Save the trained model (optional)
    model.save("bird_sound_classifier.h5")
