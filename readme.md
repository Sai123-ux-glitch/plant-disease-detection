# Plant Disease Detection using Deep Learning

# Overview

This project focuses on detecting plant diseases from leaf images using a Convolutional Neural Network (CNN).
A custom **ResNet9 architecture** is implemented using **PyTorch**, and predictions are served through a **Streamlit web application**.

The goal is to help in **early disease detection** to improve crop health and agricultural productivity.

---

##  Features

* Image-based plant disease classification
* Custom ResNet9 deep learning model
* Real-time predictions using Streamlit
* User-friendly web interface
* Supports multiple plant disease classes

---

## Model Details

* Architecture: ResNet9 (custom implementation)
* Framework: PyTorch
* Input Size: 256 × 256 RGB images
* Output: Disease class prediction
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

# Project Structure

```
project/
│
├── model.py            # Model architecture (ResNet9)
├── app.py              # Streamlit web app
├── plant_model.pth     # Trained model weights
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

# Installation & Setup

#  Clone the Repository

```
git clone https://github.com/Sai123-ux-glitch/plant-disease-detection
cd plant-disease-detection
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Application

```
streamlit run app.py
```

---

##  How to Use

1. Open the Streamlit app in your browser
2. Upload a plant leaf image
3. The model predicts the disease class
4. View prediction results instantly

---

## Dataset

* Dataset consists of labeled plant leaf images
* Includes multiple disease categories and healthy samples
* Images are preprocessed and resized to 256×256

---

##  Results

* Model achieves good accuracy on validation dataset
* Performs well on real-time image predictions
* Can be further improved with data augmentation and tuning

---

## Future Improvements

* Add Grad-CAM visualization for explainability
* Improve model accuracy with deeper architectures
* Deploy on cloud (Streamlit Cloud / AWS)
* Mobile app integration

---

##  Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

---

##  License

This project is for educational purposes.

---

##  Author

Sai Vardhan
GitHub: https://github.com/Sai123-ux-glitch
