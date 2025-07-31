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
| `file.py`,<br> `new_try.py` | Experimental scripts for testing ideas or data exploration |
| `test.py`,<br>`testing.py` | Evaluates trained model performance on separate test sets |
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
    - Saves trained weights to `trafficnet.h5`.
4. **Evaluation (`test.py`/`testing.py`)**
    - Loads model and test set.
    - Outputs accuracy, optionally confusion matrix and misclassification analysis.
5. **Prediction (`predict.py`)**
    - Loads saved model and sign label map.
    - Accepts new images, preprocesses them, predicts sign class, displays human-readable label.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy pandas pillow matplotlib
```


### 2. Prepare Data

- Download and extract a traffic sign dataset (e.g., [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)).
- Place images and CSV label files in the expected folders.


### 3. Train the Model

```bash
python train.py
```

- Adjust hyperparameters or epochs in `train.py` as desired.
- Training history will be saved as `plot.png`, model as `trafficnet.h5`.


### 4. Evaluate

```bash
python test.py
```

- Review accuracy and details in terminal output.


### 5. Predict on New Images

```bash
python predict.py --img_path example/your_image.png
```

- Prints the predicted sign class.


## 🧠 Example: Predicting a Traffic Sign

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

model = load_model('trafficnet.h5')
signnames = pd.read_csv('signnames.csv').set_index("ClassId")["SignName"].to_dict()

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
