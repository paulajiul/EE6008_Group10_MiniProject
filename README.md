# Industrial Safety - PPE Detection

## EE6008 – Deep Learning at the Edge

### Team Members:
- **Ajith Antony Shanthi** - 24219169
- **Mandisi Marshal Sibanda** - 20146817
- **Paul Aji** - 24218952
- **Sreelakshmi Krishnakumar** - 24185426
- **Sri Sai Manoj Kokkirametla** - 24194069
- **Tendai Chaka** - 20041829

**Module Leader:** Brendan Mullane

---

## Abstract:
Workplace safety is a significant concern across various industries. This project aims to enhance workplace safety through the development of a lightweight and efficient Personal Protective Equipment (PPE) detection model using OpenMV. The model focuses on identifying whether workers are wearing safety equipment, such as helmets and vests, when entering the workplace. We utilized the MobileNetV2 architecture for object detection and integrated techniques such as early stopping, model checkpoints, and centroid scoring to optimize performance, achieving an F1 score of 0.84 for non-background PPE classes.

---

## Hardware and Software Requirements:

### 1. OpenMV RT1062 Camera:
The OpenMV RT1062 camera is a small, low-power microcontroller board that simplifies the implementation of machine vision applications in real-world environments. 

### 2. Edge Impulse for Deployment:
Edge Impulse is an embedded machine learning platform supporting the MLOps pipeline for edge devices, including data collection, signal processing, model training, testing, and deployment.

![EdgeImpulseWorking](images/edgeimpulseworking.png)
Fig1. Overview of Edge Impulse (STMicroelectronics)

---

## Data Acquisition:
To build the object detection model, we used the PPE Dataset for Workplace Safety, sourced from the Roboflow Universe Platform under the Creative Commons Attribution 4.0 license. The dataset contains 1600 images of industrial and construction environments under different lighting and angles. We included additional images from the S17 dataset to diversify the data. The dataset includes three primary PPE classes: helmet, person, and vest.

---

## Preprocessing the Data:

We performed several preprocessing steps to prepare the data for the model:
1. **Loading the Datasets:** Loaded the dataset into Edge Impulse using predefined features.
2. **Reshaping and Normalizing the Data:** Split the dataset into training and test data with a validation split of 80/20, ensuring the model generalizes well on unseen data.

---

## Methodology: Model Architecture and Training

### Model Architecture:
The model is based on the **MobileNetV2** architecture, selected for its high efficiency, especially in resource-constrained environments. We utilized transfer learning with pre-trained weights from the ImageNet dataset to improve convergence and reduce model complexity. 

A custom detection head was added to MobileNetV2, consisting of two sequential 2D convolutional layers that reduce channel dimensions and generate predictions for each part of the image.

### Data Representation and Loss Function:
Bounding box annotations were converted into segmentation maps that align with the model’s output grid. We applied a weighted cross-entropy loss to prioritize detecting PPE classes (helmet, vest, and person) over the background.

### Training Process:
We used the **Adam optimizer** with a learning rate of 0.0001. To enhance efficiency and reduce overfitting, we employed:
- **Centroid Scoring:** Monitored accuracy after each epoch for better performance validation.
- **Model Checkpointing:** Saved model weights upon improvement in validation F1 score.
- **Early Stopping:** Stopped training if validation accuracy did not improve by at least 0.005 over 15 epochs, reducing computation time and overfitting.

The model was trained for 35 epochs with a batch size of 8.

---

## Performance Analysis:

The model showed strong performance in both validation and test stages:
- **Validation Results:** 
  - BACKGROUND: 100% accuracy
  - HELMET: F1 score of 0.86
  - PERSON: F1 score of 0.83
  - VEST: F1 score of 0.84

- **Test Results:** 
  - Overall accuracy: 90.57%
  - Precision: 0.91
  - Recall: 0.97
  - F1 score: 0.94

The model exhibited minimal false positives and was efficient in detecting critical objects like helmets, vests, and persons while maintaining high precision.

![F1score](images/f1scorewithmatrix.png)
Fig2. Confusion Matrix with F1 Score

![Accuracy](images/accuracytest.png)
Fig3. Test Accuracy

---

## Conclusion:

This project successfully developed a lightweight PPE detection model that can be embedded into edge devices. The model was trained to classify and detect PPE objects efficiently, with an F1 score of over 84% and test accuracy exceeding 90%. Future work could involve integrating this model with a user interface for ease of use in real-world applications.

---

## References:

- **Mullane, B.** (2025, March). Quantization Techniques. Retrieved from [Brightspace UL](https://learn.ul.ie/d2l/le/lessons/49928/topics/858239)
- **STMicroelectronics.** (n.d.). Edge Impulse. Retrieved from [Edge Impulse Overview](https://www.st.com/content/st_com/en/partner/partner-program/partnerpage/Edge_Impulse.html#:~:text=Edge%20Impulse%20is%20an%20embedded,efficiently%20on%20an%20edge%20device)
- **Universe Roboflow.** (n.d.). PPE Dataset for Workplace Safety. Retrieved from [Roboflow Universe](https://universe.roboflow.com/siabar/ppe-dataset-for-workplace-safety/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

---
