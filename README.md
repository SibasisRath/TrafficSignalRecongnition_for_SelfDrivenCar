# TrafficSignalRecognition for Self-Driven Cars

A deep learning-based, end-to-end pipeline for recognizing traffic signs, designed to support autonomous vehicles and driver-assistance systems. This project uses Convolutional Neural Networks (CNNs) to classify road signs from images and provides utility scripts for training, prediction, and evaluation.

## 🚦 Overview

This repository implements a robust traffic sign recognition system:

- **Classifies images into 43+ road sign categories**
- **Trains a custom CNN on benchmark datasets**
- **Supports rapid predictions for real-world self-driving scenarios**
- **Fully documented, modular code for easy experimentation and extension**


## 📁 Repository Structure

| File/Folder | Purpose |
| :-- | :-- |
| `trafficnet.py` | Defines the CNN model architecture for classification |
| `train.py` | Loads data, preprocesses, trains the model, visualizes training progress |
| `predict.py` | Loads trained model, predicts class of new input images |
| `signnames.csv` | Maps class indices to human-readable traffic sign names |
| `environment.yml` | Conda environment file for reproducible dependency management |
| `application.py` | Evaluates trained model performance on separate test sets |
| `trafficnet.h5` | Pre-trained network weights (binary, auto-generated after training) |
| `plot.png` | Training/validation accuracy and loss visualization |
| `example/` | Example images and/or test inputs |
| `__pycache__/` | Python bytecode cache folder (auto-generated, not for manual editing) |

## 🏗️ How Does It Work?

1. **Preprocess Data**
    - Images are loaded, resized, and normalized.
    - Dataset split into training and validation/test sets.
    - `signnames.csv` ensures class index to sign-name mapping for interpretability.
2. **Model Architecture (`trafficnet.py`)**
    - Uses a multi-layer CNN: convolutional (feature extraction), pooling (downsampling), dropout/batch norm (regularization), and dense softmax classifier.
    - Tuned for balance between accuracy and computational efficiency.
3. **Training (`train.py`)**
    - Compiles the model with Adam optimizer and categorical crossentropy loss for multi-class classification.
    - Trains the model over several epochs, logging accuracy and loss (saved as `plot.png`).
    - Saves trained weights to `trafficnet.keras`.
4. **Evaluation (`test.py`/`testing.py`)**
    - Loads model and test set.
    - Outputs accuracy, optionally confusion matrix and misclassification analysis.
5. **Prediction (`predict.py`)**
    - Loads saved model and sign label map.
    - Accepts new images, preprocesses them, predicts sign class, displays human-readable label.

## 🚀 Quick Start

### 1. Create the Conda Environment

First, ensure you have Anaconda or Miniconda installed. Then, create the environment from the `environment.yml` file. This will install all necessary dependencies in an isolated environment.

```bash
conda env create --file environment.yml --name traffic-sign-rec
```


### 2. Activate the Environment

Activate the newly created environment before running any scripts.

```bash
conda activate traffic-sign-rec
```


### 3. Prepare Data

- Download and extract a traffic sign dataset (e.g., [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)).
- Extract and place the dataset images and any label files into the appropriate project folders as expected by your scripts (e.g., a `dataset/` directory).


### 4. Train the Model

My trained model is available here : [trafficnet.keras](https://github.com/SibasisRath/TrafficSignalRecongnition_for_SelfDrivenCar/blob/main/trafficnet.keras).

So without training the model you can directly use the trained model for prediction.
Retraing the model with the same dataset and no changes in the training script will result the same.

Still if you want to do it for practice or any other reason you can do it by running the following command in the terminal.

With the `traffic-sign-rec` environment active, run the training script.

```bash
python train.py --dataset "path\to\dataset" --model trafficnet.keras --plot plot.png
```

- Adjust hyperparameters or epochs in `train.py` as desired.
- Training history will be saved as `plot.png` and the model as `trafficnet.keras`.


### 5. Evaluate and Predict

Use the evaluation and prediction scripts as needed. Ensure the environment is still active.

```bash
# Predict a single new image or a directory of images
python predict.py --model any trained model --images input directory/images --examples output directory

# run the application and try
python application.py
# make sure your web cam is on.
```


## 🧠 Example: Predicting a Traffic Sign

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

model = load_model('model name') 
signnames = pd.read_csv("Sign names csv file").set_index("ClassId")["SignName"].to_dict()

def predict_image(image_path):
    img = Image.open(image_path).resize((32,32))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = np.argmax(model.predict(img_array), axis=1)[0]
    return signnames[pred]
```


## 📝 Customization \& Experimentation

- Modify the CNN layers in `trafficnet.py` to experiment with deeper/wider networks.
- Change preprocessing routines (augmentation, normalization) to fight overfitting or improve generalization.
- Add callbacks, learning rate schedules, or regularization as needed.


## 📊 Results \& Performance

- Evaluate accuracy via provided scripts and review `plot.png` for learning curves.
- The model is capable of real-time prediction on consumer GPUs and modern CPUs.


## 🤖 Why Is This Important?

Traffic sign recognition is essential for:

- Self-driving vehicles’ perception of the environment
- Advanced Driver Assistance Systems (ADAS)
- Smart traffic analysis and road safety applications


## 📚 References

- German Traffic Sign Recognition Benchmark (GTSRB)
- Keras and TensorFlow documentation for deep learning frameworks
- Standard open-source computer vision pipelines


## 🙏 Acknowledgements

Inspired by the community and built to support the progress of autonomous vehicle research and practical applications.

