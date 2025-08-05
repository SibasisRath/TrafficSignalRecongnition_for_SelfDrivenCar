from tensorflow.keras.models import load_model
import argparse

# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m_in", "--model_input", required=True, help="path to input Keras model (.h5)")
ap.add_argument("-m_out", "--model_output", required=True, help="path to output Keras model (.keras)")
args = vars(ap.parse_args())

# Load the trained model from the old .h5 format
print("[INFO] loading model from disk...")
model = load_model(args["model_input"])

# Save the model in the new .keras format
print(f"[INFO] saving model to {args['model_output']}...")
model.save(args["model_output"])

print("[INFO] Model conversion complete.")
