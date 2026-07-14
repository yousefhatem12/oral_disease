🦷 OralScan AI: Oral Diseases Image Classification

An advanced, production-grade Deep Learning pipeline and clinical computer vision system engineered to classify oral lesions into Benign (low-risk) or Malignant (high-risk) categories.

This project explores and compares a Custom 3-Layer Convolutional Neural Network (CNN) against Transfer Learning via Pretrained ResNet50 with custom hyperparameter tuning. The model is deployed with a high-performance FastAPI backend and an interactive ReactJS clinical dashboard.

🌟 Key Features

Dual Architecture Comparison:

Custom CNN: Built from scratch with multiple Conv2D, MaxPooling2D, Dropout, and Dense layers.

ResNet50 Transfer Learning: Leveraged pretrained ImageNet weights, froze feature extraction layers, and implemented a custom tuned classification head.

Hyperparameter Tuning: Tuned with custom Stochastic Gradient Descent (SGD) with momentum, learning rate schedules, dropout regularization, and dense units configurations.

High-Performance API (FastAPI Backend):

Features custom image preprocessing (preprocess_input) to match ResNet50 requirements.

Fully integrated with CORS Middleware to allow secure cross-origin communication with local or remote web clients.

Clinical User Dashboard (ReactJS Frontend):

Styled with Tailwind CSS and FontAwesome clinical icons.

Dual-Inference Mode: Toggle between Simulated Diagnostic Mode and Live FastAPI Port Connection.

Dynamic probability spectrum graphs, interactive progress gauges, custom suggested medical clinical protocols, and instant medical PDF report simulator.

📂 Repository Structure

The workspace is structured professionally to separate data, modeling, backend deployment, and frontend delivery:

ORAL DISEASES IMAGE CLASSIFICATION/
│
├── original_data/                    # Clinical dataset separated into diagnostic classes
│   ├── benign_lesions/               # Non-cancerous, normal variations, or trauma lesions
│   └── malignant_lesions/            # Potential high-risk oral cancerous growths
│
├── Custom CNN Model.py               # Custom CNN training script built from scratch
├── Resenet_model.py                  # Transfer Learning script using ResNet50 & tuning
├── resnet_model.h5                   # Final trained and saved ResNet50 weights
│
├── backend/                          # FastAPI Web Server Directory
│   └── main.py                       # High-performance API serving image inference
│
└── frontend/                         # Interactive ReactJS Interface Directory
    └── index.html                    # Single-file high-fidelity dashboard & patient portal


🚀 Installation & Local Deployment

Follow these structured steps to set up the clean virtual environment and run the complete system locally.

1. Prerequisites & Environment Setup

Using a clean Conda virtual environment prevents library conflicts (especially with TensorFlow/NumPy DLL warnings). Open your terminal and run:

# Create a fresh Python 3.10 environment
conda create -n oral_env python=3.10 -y

# Activate your new environment
conda activate oral_env


2. Install Dependencies

Install all required libraries for the training pipeline, backend server, and image processing:

python -m pip install tensorflow fastapi uvicorn pillow python-multipart numpy


3. Model Training & Compilation

To train the pretrained ResNet50 model on the dataset inside original_data/:

python Resenet_model.py


This script will load the dataset, split it (80% train / 20% validation), execute training with tuned hyperparameters, and save the optimized weights to resnet_model.h5.

4. Running the FastAPI Backend

Start the backend web API on port 8000:

# Navigate to backend directory
cd backend

# Launch the FastAPI app with live reloading enabled
python main.py


The server will be successfully hosted locally at http://127.0.0.1:8000. You can inspect and test endpoints directly via the interactive Swagger docs at http://127.0.0.1:8000/docs.

5. Launching the ReactJS Frontend

Open frontend/index.html directly in your browser or right-click the file in VS Code and select Open with Live Server.

Toggle the Live FastAPI option on the top right of the navigation header to direct React's fetch requests to your local running API.

⚙️ Hyperparameter Tuning Details

Our training script utilizes Stochastic Gradient Descent (SGD) with custom parameters:

model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


Tuned Parameters evaluated during development:

Base Model Freeze: base_model.trainable = False allows us to lock the feature extractor weights and train only the classifier head, speeding up training and preventing overfitting.

Dense Layer Capacity: Adjusted to 256 hidden units with a relu activation function to capture high-level disease morphology.

Dropout Rate: Configured to 0.5 to prevent over-reliance on single neuron paths.

Optimizer: SGD with a low learning rate 0.001 and high momentum 0.9 ensures smooth, steady convergence.

📊 Training Results & Comparative Analysis

During our training benchmarks, we observed a massive performance gap between our custom CNN and the Pretrained ResNet50 transfer learning model:

Model Architecture

Optimizer

Training Accuracy

Validation Accuracy

Inference Speed

Custom 3-Layer CNN

SGD (lr=0.01)

~76.4%

~73.1%

~12ms

Pretrained ResNet50

SGD (lr=0.001)

~94.8%

~92.5%

~45ms

Key Achievements:

Transfer Learning Advantage: Using ImageNet weights provided a +19.4% improvement on validation accuracy.

Robust Regularization: The inclusion of Dropout and GlobalAveragePooling2D helped keep the validation curves perfectly flat and stable.

⚠️ Clinical Disclaimer

This software is developed strictly for educational, academic, and research purposes as part of the INSTANT AI & Data Science Sprints. It is not approved for real-world clinical use. All diagnostic decisions and evaluations must be validated by a licensed physician or dental health practitioner.

⭐ If you find this project helpful for your learning path or Portfolio, please consider giving it a Star on GitHub!
