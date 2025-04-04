# Facial Verification System

## Project Overview
This facial verification system uses deep learning to verify a person's identity by comparing facial characteristics. The system creates high-dimensional embeddings of facial images and determines similarity through distance measurements, providing a robust way to authenticate individuals in real-time.

## Features
- Embedding-based facial verification using convolutional neural networks
- Real-time verification through webcam integration
- Flask web application for easy user interaction
- High precision and recall (1.0 on test dataset)
- Customizable verification thresholds

## Technical Architecture

### Data Collection & Organization
The system was trained using three types of datasets:
- **Anchor dataset**: Core images of target individuals captured via OpenCV
- **Positive dataset**: Additional images of the same individuals for training verification
- **Negative dataset**: Images of different individuals collected from the internet

### Model Architecture
The verification model consists of:
1. **Embedding Network**: CNN-based architecture that transforms facial images into 4096-dimensional vector embeddings
2. **L1 Distance Layer**: Calculates the Manhattan distance between anchor and validation image embeddings
3. **Decision Layer**: Dense layer with sigmoid activation that determines similarity probability

### Verification Process
The system follows these steps for verification:
1. Captures user's image through the webcam
2. Processes the image through the embedding network
3. Compares the resulting embedding with each validation image
4. Applies a detection threshold (50%) to determine matches
5. Applies a verification threshold (50% of total validation images)
6. Returns verification result to the user

## Project Structure
```
Facial-Verification-Project/
├── datasets/
│   ├── anchor/         # Target individual images
│   ├── positive/       # Additional target images
│   └── negative/       # Non-target individual images
├── models/
│   └── siamese_model/  # Trained model files
├── application/
│   ├── Verification_images/  # Reference images for verification
│   └── input_image/        # Temporary storage for captured images
├── static/             # Web app static assets
├── templates/          # Flask HTML templates
└── app.py /             # Flask web application
└── requirements.txt /

```

## Implementation Details

### Dataset Creation
- Combined datasets using TensorFlow's data API
- Created positive pairs with labels of 1: `tf.data.dataset.zip(anchor, positive, tf.data.dataset.tensor_from_slice(tf.ones))`
- Created negative pairs with labels of 0: `tf.data.dataset.zip(anchor, negative, tf.data.dataset.tensor_from_slice(tf.zeros))`

### Training
- Model trained for 50 epochs
- Used binary crossentropy loss and Adam optimizer
- Achieved perfect precision and recall on test dataset

### Verification Thresholds
- **Detection Threshold**: 50% similarity for individual image comparison
- **Verification Threshold**: Requires 50% of validation images to match for successful verification

## Setup and Usage

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Flask
- NumPy

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Facial-Verification-Project.git
cd Facial-Verification-Project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Application
```bash
python src/app.py
```
Access the application at `http://localhost:5000` in your web browser.

### Using the Verification System
1. Navigate to the web interface
2. Position your face in the camera frame
3. Click "Capture Image" to take a photo
4. The system will process your image and display the verification result
5. A "Verified" message appears if you match the validation images above the threshold

## Future Improvements
- Integration with access control systems
- Mobile application development
- Enhanced preprocessing for varying lighting conditions
- Face alignment for improved accuracy
- Liveness detection to prevent spoofing attacks

## License
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
