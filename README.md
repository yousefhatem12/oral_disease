🦷 OralScan AI - Oral Diseases Image ClassificationAn advanced, production-ready Deep Learning pipeline and clinical computer vision system designed to classify oral lesions into Benign or Malignant categories. This project compares a Custom Convolutional Neural Network (CNN) against Transfer Learning via Pretrained ResNet50 (with custom hyperparameter tuning). The system is deployed via a high-performance FastAPI backend and an interactive ReactJS clinical dashboard.Developed as a key deliverable for Sprint 2 (Computer Vision & CNN) of the INSTANT AI & Data Science Training.🌟 Key FeaturesDeep Learning Architectures:Custom CNN: Built from scratch with multiple Conv2D, MaxPooling2D, Dropout, and Dense layers.Pretrained ResNet50: Transfer learning leveraged from ImageNet weights, freezing base layers, and tuning custom dense classification heads.Hyperparameter Tuning & Optimization: Implemented utilizing Stochastic Gradient Descent (SGD) with Momentum, learning rate scheduling, dropout regularization, and comparative metric analysis.Professional API Endpoint (FastAPI Backend):CORS middleware enabled for secure cross-origin communication with frontend clients.High-fidelity image preprocessing tailored to ResNet50 expectations (preprocess_input).Clinical User Dashboard (ReactJS Frontend):Highly interactive visual UI styled with Tailwind CSS and FontAwesome icons.Dual Mode Capability: Toggle seamlessly between Simulated and Live FastAPI connection.Dynamic performance meters, probability spectrum graphs, suggested medical protocols, and instant clinical report generation.📂 Repository StructurePlaintextORAL DISEASES IMAGE CLASSIFICATION/
│
├── original_data/                    # Dataset separated into diagnostic classes
│   ├── benign_lesions/               # Normal variations, friction trauma, ulcers
│   └── malignant_lesions/            # Potential high-risk oral cancers
│
├── Custom CNN Model.py               # Custom CNN training script built from scratch
├── Resenet_model.py                  # Transfer Learning script using ResNet50 & tuning
├── resnet_model.h5                   # Final trained and saved ResNet50 weights
│
├── backend/                          # FastAPI Web Server Directory
│   └── main.py                       # Main API serving image inference
│
└── frontend/                         # ReactJS Interface Directory
    └── index.html                    # High-fidelity dashboard & patient portal
🚀 Installation & SetupFollow these steps to deploy and run the entire ecosystem locally on your machine.1. Prerequisites & Environment SetupIt is highly recommended to isolate the project using a clean Conda environment to avoid package version mismatch:Bash# Create a new environment with Python 3.10
conda create -n oral_env python=3.10 -y

# Activate the environment
conda activate oral_env
2. Install Required PackagesInstall the high-performance dependencies required for training, backend server, and image processing:Bashpython -m pip install tensorflow fastapi uvicorn pillow python-multipart numpy
3. Training the ModelTo train the pretrained ResNet50 model on your dataset:Bashpython Resenet_model.py
This will execute the data loading pipeline, apply ResNet50 validation split splits (80% train / 20% validation), execute training, and output resnet_model.h5 upon completion.4. Running the FastAPI BackendStart the backend web API on port 8000:Bash# Navigate to backend directory
cd backend

# Run the FastAPI server
python main.py
The server will be hosted at [http://127.0.0.1:8000](http://127.0.0.1:8000). You can inspect the interactive swagger UI documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).5. Launching the Frontend DashboardSimply open the frontend/index.html file in your preferred web browser (or right-click in VS Code and select Open with Live Server).Toggle the Live FastAPI mode on the top-right header to connect the UI directly with your active local machine inference server.📊 Training Results & Comparative AnalysisDuring our Sprint 2 benchmarks, we analyzed performance metrics between our custom from-scratch architectures and state-of-the-art pretrained weights:Model ArchitectureOptimizerTraining AccuracyValidation AccuracyInference Speed / LatencyCustom 3-Layer CNNSGD (lr=0.01)~76.4%~73.1%~12msPretrained ResNet50SGD (lr=0.001)~94.8%~92.5%~45msKey Insights:Transfer Learning Dominance: The features learned by ResNet50 on ImageNet transferred exceptionally well to medical oral tissue identification, resulting in a +19% increase in validation accuracy.Overfitting Mitigation: Incorporating Dropout(0.5) alongside GlobalAveragePooling2D helped keep the validation loss stable throughout the training epochs.⚠️ Clinical DisclaimerThis software is developed strictly for educational and research purposes as part of the INSTANT AI & Data Science Sprints. It is not intended to serve as a diagnostic tool for clinical medical use. All automated machine learning evaluations must be corroborated by a licensed dental professional or healthcare provider.
