## ProjectDIP - Hand Gesture Recognition

This project focuses on real-time hand gesture recognition using the OpenCV library and a pre-trained deep learning model. The goal is to detect and recognize hand gestures from a live video stream captured by a webcam. The project consists of two main files, `projectdip.py` and `test.py`, each serving different purposes.

### Requirements:

- Python 3.x
- OpenCV (cv2)
- cvzone
- NumPy
- TensorFlow (for loading and using the pre-trained model)
- Webcam (built-in or external)

### Installation:

1. Clone or download the repository to your local machine:

```
git clone <repository_url>
```

2. Install the required packages using pip (make sure you're in the project directory):

```
pip install -r requirements.txt
```

### Usage:

1. Run `projectdip.py` to start capturing the live video stream, detect hand gestures, and save processed gesture images. Press the "s" key to save the gesture image when a hand is detected.

```
python projectdip.py
```

2. Run `test.py` to start capturing the live video stream, detect hand gestures, and recognize gestures using the pre-trained classification model. Press the "q" key to save the gesture image and exit the program.

```
python test.py
```

### Model Information:

The pre-trained classification model (`keras_model.h5`) used in the `test.py` script was generated using Teachable Machine, a tool provided by Google. The model was trained to recognize hand gestures based on a set of labeled training images. The corresponding label file (`labels.txt`) provides the mapping between the model's predicted indices and the actual gesture labels.

### Acknowledgments:

This project utilizes the `HandTrackingModule` and `ClassificationModule` from the `cvzone` library to achieve hand tracking and gesture recognition functionalities. The pre-trained classification model and the corresponding label file, generated using Teachable Machine by Google, are essential components of the recognition process.

Note: Replace file paths (such as folder paths for saving images and model/label paths) with your actual file paths.

Please ensure you have the necessary dependencies and files correctly set up before running the code. Make sure your webcam is connected and accessible for capturing video streams.
