# Sugar-Cane-disease-classification-using-CNN

**Sugarcane Disease Classification Using CNN**

**Description:**
This project aims to develop a Convolutional Neural Network (CNN) model for the classification of sugarcane diseases. The dataset consists of images of sugarcane leaves infected with various diseases, along with healthy sugarcane leaves. The goal is to train a CNN model to accurately classify the images into their respective disease categories or label them as healthy.

**Project Goal:**
The primary goal of this project is to build an efficient CNN model that can accurately classify sugarcane leaf images into different disease categories, enabling early detection and diagnosis of sugarcane diseases. By automating the classification process, the model can assist farmers in identifying diseased plants promptly, allowing them to take timely remedial actions to mitigate crop losses. Additionally, the project aims to enhance the model's performance by incorporating data augmentation techniques such as rotation, zoom, flip, as well as optimizing the data pipeline using caching and prefetching mechanisms to expedite the training process and improve overall efficiency.

**Key Steps:**
1. **Loading and Preprocessing the Dataset:** The dataset containing images of sugarcane leaves is loaded and preprocessed, including resizing, normalization, and data augmentation to increase the diversity of the training data.

2. **Building the CNN Model:** A Convolutional Neural Network (CNN) architecture is constructed using TensorFlow/Keras. The model consists of convolutional layers, max-pooling layers, and dense layers, followed by a softmax activation function in the output layer for multiclass classification.

3. **Compiling and Training the Model:** The CNN model is compiled with appropriate loss and optimization functions, and then trained on the preprocessed dataset. Training is conducted for a specified number of epochs, with periodic evaluation of performance metrics such as accuracy and loss.

4. **Model Evaluation:** Once training is complete, the trained model is evaluated using a separate test dataset to assess its performance on unseen data. Metrics such as accuracy, precision, recall, and F1-score are calculated to measure the model's effectiveness in classifying sugarcane diseases.

5. **Inference and Prediction:** The trained model is deployed to make predictions on new or unseen images of sugarcane leaves. Inference is performed by feeding the images through the trained model, which outputs the predicted disease category or indicates if the leaf is healthy.

**Conclusion:**
The project demonstrates the potential of CNN models in accurately classifying sugarcane diseases, thereby aiding farmers in early detection and management of plant diseases. By leveraging advanced deep learning techniques and data augmentation strategies, the model achieves robust performance and can be deployed in real-world agricultural applications to improve crop health and productivity.
