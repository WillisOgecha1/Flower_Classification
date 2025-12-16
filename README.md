🌸 CNN-Based Flower Image Classification
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for multi-class flower image classification using deep learning techniques. The objective is to accurately classify flower images into their respective categories by learning hierarchical visual features such as edges, textures, and shapes.

The project demonstrates an end-to-end computer vision workflow, including data loading, preprocessing, model design, training, evaluation, and performance analysis.

🎯 Problem Statement

Image classification is a core task in computer vision with applications in agriculture, botany, e-commerce, and biodiversity research.
This project addresses the problem of automatically identifying flower species from images using a CNN trained on labeled flower images.

📂 Dataset

Type: Image dataset (RGB images)

Classes: Multiple flower categories (e.g., daisy, dandelion, rose, sunflower, tulip)

Structure:

flower_photos/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/


Each subfolder represents a class label.

🧠 Model Architecture

The CNN architecture is designed to progressively extract spatial features from the images:

Convolutional layers with ReLU activation

MaxPooling layers for spatial down-sampling

Fully connected (Dense) layers for classification

Softmax activation in the output layer

Key components:

Feature extraction using convolution filters

Dimensionality reduction via pooling

Non-linear learning using activation functions

Multi-class classification using Softmax

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

OpenCV / PIL

Jupyter Notebook

🔄 Workflow

Load and explore the dataset

Image preprocessing and resizing

Dataset splitting (training / validation)

CNN model construction

Model compilation

Training and validation




📊 Model Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Evaluation Metric: Accuracy

Training performed over multiple epochs with validation monitoring

📈 Results

The trained model successfully learns discriminative features from flower images and achieves strong classification performance on the validation set.

Performance evaluation includes:

Training vs validation accuracy

Training vs validation loss

Visual inspection of predictions


3️⃣ Install Dependencies
pip install tensorflow numpy matplotlib opencv-python pillow

4️⃣ Run the Notebook
jupyter notebook cnn_Flower_image_Classification.ipynb

📁 Project Structure
├── cnn_Flower_image_Classification.ipynb
├── datasets/
│   └── flower_photos/
├── README.md
└── requirements.txt

🔍 Key Learnings

Practical implementation of CNNs for image classification

Image preprocessing and directory-based labeling

Handling multi-class classification problems

Monitoring overfitting using validation curves


👤 Author

Willis Ogecha
Machine Learning & AI Enthusiast
Background in Software Engineering, Data Science, and Applied ML

⭐ Acknowledgments

TensorFlow and Keras documentation

Open-source flower image datasets

Deep learning community resources
