# Leakage Detection in Vacuum Bagging using Neural Networks

## Overview
This project aims to develop a neural network-based model for detecting leakage coordinates in vacuum bagging processes. The model predicts leakage coordinates from normalized flow rates, incorporating prior knowledge through equivariance conditions to improve accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Equivariance Conditions](#equivariance-conditions)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Introduction
In vacuum bagging processes, detecting leakage accurately is crucial. The project focuses on predicting the coordinates of leakages using a parameterized function \( f_θ \). The model is trained on data where the fourth sensor (MFC4) had a malfunction, causing higher values than the other sensors. The project explores both standard neural network architectures and those incorporating equivariance conditions.

## Dataset
The project uses two training datasets:
- `leakage_dataset_train_100.csv` (100 samples)
- `leakage_dataset_train_1000.csv` (1000 samples)

Each dataset contains normalized flow rates and corresponding leakage coordinates.

## Methodology
The project involves:
1. **Data Preprocessing**: Handling sensor malfunctions and normalizing data.
2. **Network Architecture**: Implementing both standard and equivariant neural network architectures.
3. **Training**: Training the models on both datasets with and without data augmentation.
4. **Evaluation**: Comparing the performance of the models using validation and test datasets.

## Equivariance Conditions
The model incorporates prior knowledge through equivariance conditions. These conditions ensure that for any given set of flow rates, the predicted leakage coordinates remain consistent under rotations and flips. This is achieved through:
- Rotations by 90°, 180°, and 270°
- Vertical flips
- Combinations of rotations and flips

This enhances the model's robustness and accuracy.

## Usage
To run the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Arjunarora/Leakage_Detection_Vaccum_Bag_NN.git
    cd Leakage_Detection_Vaccum_Bag_NN
    ```

2. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```sh
    jupyter notebook Leakage_detection_final.ipynb
    ```

4. **Train the model**:
    - Follow the steps in the notebook to preprocess the data, define the network architecture, and train the model.

## Results
The performance of the models is evaluated based on their accuracy in predicting leakage coordinates. The equivariant model, with its additional prior knowledge, is expected to outperform the standard model, especially in scenarios with limited training data.

## Future Work
Future improvements could include:
- Testing with larger datasets.
- Further optimizing network architectures.
- Applying the model to real-world vacuum bagging systems.

## Contributors
- **Arjun Arora** (Project Lead)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
