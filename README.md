# Customer Support Ticket Classification Using Deep Learning

## 1. Introduction

Customer support systems receive thousands of tickets daily, making manual routing inefficient and time-consuming. This project focuses on automating ticket classification using Deep Learning models such as RNN, LSTM, Bi-Directional LSTM, and GRU. The system predicts the appropriate support queue based on the ticket content and further integrates with the Gemini API to generate polite automated responses for customers.

The project uses the Hugging Face dataset Hugging Face Tobi-Bueck/customer-support-tickets, where the customer ticket body is used as input and the queue label is used as the prediction target.

## 2. Dataset Preparation

The dataset was loaded from the Hugging Face datasets library using the customer support ticket dataset.

Dataset Details
Dataset Name: Tobi-Bueck/customer-support-tickets
Input Feature: body
Contains the textual content of customer support tickets.
Target Feature: queue
Represents the department or support queue responsible for resolving the issue.
Objective

The primary goal of the project is to classify incoming customer tickets into the correct support queue automatically.

Example Workflow
Customer submits a support ticket.
The system reads the ticket body.
The trained deep learning model predicts the correct queue.
Gemini API generates a professional acknowledgment response.

## 3. Data Preprocessing

Text preprocessing is a crucial step for improving model performance and reducing noise in the dataset.

## 3.1 Text Cleaning

The ticket bodies were cleaned using several preprocessing techniques:

Conversion to lowercase
Removal of punctuation marks
Removal of special characters
Elimination of extra spaces
Handling missing or null values

This step ensures consistency across all ticket texts.

## 3.2 Tokenization

The cleaned text data was tokenized to convert textual information into numerical sequences understandable by neural networks.

Tokenization Process
Vocabulary creation from ticket text
Conversion of words into integer indices
Rare words handled using out-of-vocabulary tokens
3.3 Padding and Truncation

Since customer tickets vary in length, sequence standardization was necessary.

Padding
Shorter sequences were padded with zeros.
Truncation
Longer sequences were truncated to a fixed maximum length.

This allowed all input sequences to have uniform dimensions suitable for neural network training.

## 3.4 Label Encoding

The target column queue contains categorical labels representing different departments.

Encoding Method
Each queue name was converted into a numerical label using label encoding.

Example:

Billing → 0
Technical Support → 1
Account Management → 2

This enabled the classification models to process target labels efficiently.

## 4. Model Development

Several recurrent neural network architectures were implemented and compared.

### 4.1 Recurrent Neural Network (RNN)

A basic Many-to-One RNN architecture was developed for sequence classification.

Architecture
Embedding Layer
Simple RNN Layer
Dense Output Layer
Softmax activation for multi-class classification
Characteristics
Captures sequential dependencies
Suffers from vanishing gradient problems for long sequences
Accuracy

0.2354

### 4.2 Long Short-Term Memory (LSTM)

An LSTM model was implemented to overcome the limitations of traditional RNNs.

Architecture
Embedding Layer
LSTM Layer
Dense Output Layer
Softmax activation
Advantages
Better memory handling
Captures long-term dependencies
Improved contextual understanding
Accuracy

0.2830

### 4.3 Bi-Directional LSTM

A Bi-Directional LSTM model processes text sequences in both forward and backward directions.

Architecture
Embedding Layer
Bi-Directional LSTM Layer
Dense Layer
Softmax activation
Advantages
Learns context from both directions
Improves semantic understanding
Better classification performance
Accuracy

0.4280

### 4.4 Gated Recurrent Unit (GRU)

A GRU model was also developed for comparison.

Architecture
Embedding Layer
GRU Layer
Dense Output Layer
Softmax activation
Advantages
Faster training compared to LSTM
Fewer parameters
Effective sequence learning
Accuracy

0.4297

## 5. Model Evaluation

The models were evaluated using standard classification metrics.

Evaluation Metrics
Accuracy
Precision
Recall
F1-Score

These metrics help assess classification quality and overall predictive performance.

## 6. Performance Comparison
Model	Accuracy
RNN	0.2354
LSTM	0.2830
Bi-Directional LSTM	0.4280
GRU	0.4297

## 7. Analysis of Results

The experimental results show that advanced recurrent architectures significantly outperform the basic RNN model.

Observations
The traditional RNN achieved the lowest accuracy due to difficulty handling long-term dependencies.
LSTM improved performance by preserving contextual memory.
Bi-Directional LSTM further enhanced prediction quality by learning from both past and future contexts.
GRU achieved the highest accuracy while maintaining computational efficiency.
Best Performing Model

The GRU model achieved the highest classification accuracy of 42.97%, slightly outperforming the Bi-Directional LSTM model.

## 8. Gemini API Integration

After predicting the queue, the project integrates with Google Gemini API to generate automated customer responses.

Workflow
Customer ticket is submitted.
Deep learning model predicts the support queue.
Ticket text and predicted queue are passed to Gemini API.
Gemini generates a polite acknowledgment reply.
Example Generated Response
Customer Ticket

“I am unable to access my account after resetting my password.”

Predicted Queue

Technical Support

Gemini Response

“Thank you for contacting support. We understand that you are experiencing issues accessing your account after resetting your password. Our Technical Support team has received your request and will assist you shortly. We appreciate your patience.”

## 9. Advantages of the System
Automated ticket routing
Reduced manual workload
Faster response handling
Improved customer experience
Intelligent acknowledgment generation
Scalable support management solution
10. Limitations
Moderate classification accuracy due to dataset complexity
Imbalanced queue distribution may affect predictions
Limited contextual understanding for highly complex tickets
Requires larger datasets for improved performance

## 12. Conclusion

This project successfully demonstrates the use of deep learning techniques for customer support ticket classification. Multiple recurrent architectures were implemented and evaluated on a real-world support ticket dataset. Among all models, the GRU and Bi-Directional LSTM models delivered the best performance.

The integration with Gemini API further enhanced the system by automatically generating professional customer acknowledgment messages. Overall, the project highlights the practical application of Natural Language Processing and Deep Learning in automating customer support operations.
