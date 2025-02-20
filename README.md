# Melanoma Detection Assignment  

## Table of Contents  
- **Problem Statement**  
- **Objectives**  
- **Approach**  
- **Technologies/Libraries Used**  
- **Conclusions**  
- **Acknowledgements**  
- **Glossary**  
- **Author**  

---

## Problem Statement  
The goal of this project is to develop a **custom CNN-based model** for melanoma detection. **Melanoma** is a highly aggressive form of skin cancer responsible for **75% of skin cancer-related deaths**. Early detection is critical, and an AI-driven solution that can analyze images and alert dermatologists to potential cases of melanoma can significantly reduce the manual effort required for diagnosis.  

The dataset used for this project consists of **2,357 images** of both malignant and benign oncological skin diseases. The images were sourced from the **International Skin Imaging Collaboration (ISIC)** and categorized based on their respective classifications. The dataset includes various skin conditions, with **melanomas and moles** being slightly more dominant in quantity.  

### The dataset includes the following skin diseases:  
- Actinic keratosis  
- Basal cell carcinoma  
- Dermatofibroma  
- Melanoma  
- Nevus  
- Pigmented benign keratosis  
- Seborrheic keratosis  
- Squamous cell carcinoma  
- Vascular lesion  

### **Important Guidelines:**  
- The model must be **built from scratch** without using **transfer learning** or pre-trained models.  
- Some new concepts are introduced in this assignment, but necessary guidance is provided to ensure smooth learning.  
- Due to the large dataset and high number of epochs, **training the model may take time**. It is recommended to use a **GPU runtime on Google Colab** for efficient execution.  

---

## Objectives  
The primary goal of this assignment is to design and implement a **custom convolutional neural network (CNN)** to accurately classify different types of skin lesions, with a focus on detecting **melanoma**. The model aims to:  
- Assist dermatologists in early melanoma detection by automating image analysis.  
- Reduce the manual effort required for identifying various skin conditions.  
- Explore different CNN architectures, regularization techniques, and optimization methods to improve classification accuracy and robustness.  

---

## Approach  
1. **Import Necessary Libraries**  
2. **Load and Understand the Dataset**  
3. **Data Preprocessing:** Split the dataset into training, validation, and test sets.  
4. **Build the Initial CNN Model (Model 1)**  
5. **Implement Data Augmentation and Train an Enhanced Model**  
6. **Address Class Imbalance and Retrain the Model**  
7. **Fine-Tune the Model for Optimal Performance**  
8. **Make Predictions Using the Trained Models**  
9. **Evaluate and Conclude Findings**  

---

## Technologies/Libraries Used  
- **numpy** : 1.26.4  
- **pandas** : 2.2.2  
- **matplotlib** : 3.7.1  
- **tensorflow** : 2.17.0  
- **keras** : 3.4.1  
- **augmentor** : 0.2.12  

---

## Conclusions  
After training and evaluating multiple CNN models, the following key insights were observed:  
- **Model 9**, which utilized **ten convolutional layers**, **data augmentation**, **balanced class distribution**, **dropouts**, **batch normalization**, and a **controlled learning rate**, demonstrated the best overall performance. It achieved a **validation accuracy of 0.73** and a **test accuracy of 0.50**, making it the most generalizable model.  
- **Models 4 and 5** performed moderately well, with validation accuracies of **0.70 and 0.65** and test accuracies around **0.45**. These results highlight the significance of deeper architectures and regularization techniques.  
- **Base Models (1, 2, 3)** suffered from severe **overfitting**, particularly **Model 1**, which had a **training accuracy of 0.91** but a **very low test accuracy of 0.33**. This reinforces the importance of using **dropouts, batch normalization, and data augmentation** to improve generalization.  
- Increasing the number of epochs does not always enhance performance. **Model 10**, trained for additional epochs, showed better training accuracy but did not significantly improve test accuracy.  

### **Key Takeaways:**  
- Deeper CNN architectures with **controlled learning rates, batch normalization, and dropout regularization** significantly improved model performance.  
- **Model 9** was the most balanced, providing the best trade-off between validation and test accuracy.  
- Further fine-tuning and hyperparameter adjustments can help refine the model for even better melanoma detection.  

---

## Q&A Section  
- **Which class has the least number of samples?**  
  - *Answer:* **Seborrheic keratosis**  
- **Which class dominates the dataset in terms of proportionate samples?**  
  - *Answer:* **Pigmented benign keratosis**  

---

## Glossary  
- **Data Augmentation**  
- **Class Imbalance**  
- **Train-Validation Split**  
- **Test Set**  
- **Convolutional Neural Network (CNN)**  
- **Dropout**  
- **Learning Rate (LR)**  
- **Overfitting**  
- **Early Stopping**  
- **Cross-Entropy Loss**  
- **Accuracy**  
- **Batch Normalization**  
- **Max Pooling**  
- **Softmax**  
- **Learning Rate Scheduler (ReduceLROnPlateau)**  
