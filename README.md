# Hespress-Classification
The code goes through the following training process:

1. **Data Preprocessing**: The code applies several preprocessing steps to the text data, including stemming, stop word removal, and normalization of Arabic characters.

2. **Data Preparation**: The data is loaded from CSV files and preprocessed using the `prepareDataSet` function. It combines the 'score' and 'comment' columns from DataFrames and splits the data into training and testing sets using a train-test split.

3. **Feature Extraction**: The code uses TF-IDF vectorization to convert text data into numerical features. It extracts the TF-IDF features from the training and testing sets using `TfidfVectorizer`.

4. **Dimensionality Reduction**: To reduce the dimensionality of the TF-IDF features, the code applies TruncatedSVD, transforming the TF-IDF features into a lower-dimensional representation.

5. **Model Definition**: The neural network model is defined using the Sequential API from Keras. It consists of multiple hidden layers with a ReLU activation function and a final output layer with a sigmoid activation function for binary classification.

6. **Model Compilation**: The model is compiled using the 'adam' optimizer and 'binary_crossentropy' loss function for binary classification. The metric 'accuracy' is used to monitor the performance during training.

7. **Model Training**: The model is trained on the training data using the 'fit' function. Early stopping is implemented as a callback to stop training if the model's validation performance does not improve.

8. **Model Evaluation**: After training, the model is evaluated on the test data using the 'evaluate' function to calculate the loss and accuracy of the model on the test set.

9. **Additional Metrics**: The code makes predictions on the test set and converts the predicted probabilities to binary class labels. Then, it calculates precision, recall, and F1-score using the predicted labels and true labels. The metrics are computed using functions from `sklearn.metrics`.

10. **Print Metrics**: Finally, the code prints the metrics, including accuracy, precision, recall, and F1-score, to assess the performance of the trained model.

**Meaning of Each Result and Metric**:
For the Hespress data, the results in the output mean the following:

1. **Test Loss (0.4758664071559906)**: The test loss represents the average value of the binary cross-entropy loss function on the test set. A lower test loss indicates that the model's predicted probabilities are closer to the true binary labels. In this case, the test loss is relatively low, suggesting that the model is performing well in predicting the target classes.

2. **Test Accuracy (0.795852541923523)**: Test accuracy is the proportion of correctly classified instances in the test set. It shows the overall accuracy of the model in binary classification. In this case, the test accuracy is approximately 79.59%, indicating that the model correctly predicts the target classes for about 79.59% of the instances in the test set.

3. **Precision (0.805444369063772)**: Precision is the ratio of true positive predictions to the total positive predictions made by the model. It indicates the ability of the model to avoid false positives. In this case, the precision is approximately 80.54%, meaning that about 80.54% of the instances that the model predicts as positive are actually positive.

4. **Recall (0.9743931715124033)**: Recall, also known as sensitivity or true positive rate, is the ratio of true positive predictions to the total number of actual positive instances in the test set. It shows the ability of the model to correctly identify positive instances. In this case, the recall is approximately 97.44%, meaning that the model correctly identifies about 97.44% of the actual positive instances in the test set.

5. **F1-score (0.8819001457793624)**: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall and is useful when there is an imbalance between the number of positive and negative instances in the data. A higher F1-score indicates better model performance. In this case, the F1-score is approximately 88.19%, suggesting that the model has a good balance between precision and recall.

Overall, the results indicate that the model is performing well in predicting the binary classes, with high recall and relatively good precision. 
Thanks for reading :)
