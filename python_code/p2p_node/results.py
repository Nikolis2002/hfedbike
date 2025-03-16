import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("final_model_node_A.h5")  # Adjust filename if needed

# Test inputs (same as used during training)
input1 = np.array([[0], [1], [0], [1]], dtype=np.int32)
input2 = np.array([[0], [1], [1], [0]], dtype=np.int32)

# Expected labels
expected_labels = np.array([[0, 1, 0],  # Expected for first input
                            [1, 0, 0],  # Expected for second input
                            [0, 0, 1],  # Expected for third input
                            [0, 0, 1]]) # Expected for fourth input

# Make predictions
predictions = model.predict({'fruit1': input1, 'fruit2': input2})

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
expected_classes = np.argmax(expected_labels, axis=1)

# Print results
for i, (pred, exp) in enumerate(zip(predicted_classes, expected_classes)):
    print(f"Input {i+1}: Predicted Class {pred} | Expected Class {exp}")

# Check accuracy
correct_predictions = np.sum(predicted_classes == expected_classes)
accuracy = correct_predictions / len(expected_classes) * 100
print(f"\nModel Accuracy on Test Inputs: {accuracy:.2f}%")
