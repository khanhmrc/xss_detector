import pandas as pd
import numpy as np
from keras.models import load_model

def add_labels_to_data(model_dir, data_file):
    # Load the model
    model = load_model(model_dir)

    # Read the data from the CSV file
    data = pd.read_csv(data_file)

    # Extract the payloads from the data
    payloads = data.iloc[:, 0].tolist()

    # Convert the payloads to numerical input for the model
    input_data = np.array(payloads)

    input_data = input_data.reshape(-1, 1)
    # Reshape the input data if needed
    # input_data = input_data.reshape(-1, 1)  # Uncomment this line if the input shape needs to be reshaped

    # Predict labels for the input data using the model
    predicted_labels = model.predict(input_data)

    # Round the predicted labels to 0 or 1
    rounded_labels = np.round(predicted_labels)

    # Add the predicted labels to the data
    data['Label'] = rounded_labels

    # Save the updated data to a new CSV file
    new_file = data_file.replace('.csv', '_labeled.csv')
    data.to_csv(new_file, index=False)

    print("Labels added to the data and saved to", new_file)

if __name__ == "__main__":
    model_dir = "G:/file/MLP_model.h5"
    data_file = "data/mutated_payloads.csv"

    add_labels_to_data(model_dir, data_file)