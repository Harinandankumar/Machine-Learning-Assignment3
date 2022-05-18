#!/usr/bin/env python
# coding: utf-8

# 1. Explain the term machine learning, and how does it work? Explain two machine learning
#    applications in the business world. What are some of the ethical concerns that machine learning
#    applications could raise?
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#       Machine Learning is a form of artificial intelligence (AI) that teaches computers to think in a similar way to how humans do: Learning and improving upon past experiences. It works by exploring data and identifying patterns, and involves minimal human intervention.
# 
# There are various applications in Business World :
# 
# 1. Real-time chatbot agents.
# 2. Decision support.
# 3. Customer recommendation engines.
# 4. Customer churn modeling.
# 5. Dynamic pricing tactics.
# 6. Market research and customer segmentation.
# 7. Fraud detection.
#   Also there are some of the ethical concerns that ML applications could raise : AI presents three major areas of ethical concern for society: Privacy and surveillance, bias and discrimination, and perhaps the deepest, most difficult philosophical question of the era, the role of human judgment.

# 2. Describe the process of human learning:
# i. Under the supervision of experts
# 
# ii. With the assistance of experts in an indirect manner
# 
# iii. Self-education
# 
# 
# 
# 
# 
# 
# Ans:- 
#      The processes of Human learning are described below:
# 
#  Under the supervision of experts: Human-guided machine learning is a process whereby subject matter experts accelerate the  learning process by teaching the technology in real-time. For example, if the machine learning model comes across a piece of data it is uncertain about, a human can be asked to weigh in and give feedback. The model then learns from this input, and uses it to make a more accurate prediction the next time. Human-guided machine learning works from the bottom up by first using algorithms to conduct the heavy lifting of identifying relationships within the data, and engaging humans when necessary for training or validation Concept Learning.
# 
#  With the assistance of experts in an indirect manner: Well The process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance Operant Conditioning.
# 
#  Self education: Ability to recognize patterns, learn from data, and become more intelligent over time (can be AI or programmatically based).Machine Learning: AI systems with ability to automatically learn and improve from experience without being explicitly programmed via training Hebbian Learning.  

# 3. Provide a few examples of various types of machine learning.
# 
# 
# 
# 
# 
# 
# Ans:-
#        Examples of Various types of Machine Learning techniques are:
# 
# Example of Supervised Learning is text classification problems. In this set of problems, the goal is to predict the class label of a given piece of text. One particularly popular topic in text classification is to predict the sentiment of a piece of text, like a tweet or a product review,Image segmentation, Medical Diagnosis
# 
# Example of Unsupervised Learning : Fraud detection, Malware detection, Anomaly detection, Clustering Analysis, Identification of human errors during data entry Conducting accurate basket analysis, etc.
# 
# Example of Reinforcement Learning : Applications in self-driving cars, Industry automation : learning-based robots are used to perform various tasks.

# 4. Examine the various forms of machine learning.
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#        These are three types of Machine Learning Techniques:
# 
# 1. Supervised Learning
# 2. Unsupervised Learning
# 3. Reinforcement Learning.

# 5. Can you explain what a well-posed learning problem is? Explain the main characteristics that must
#     be present to identify a learning problem properly.
# 
# 
# 
# 
# 
# 
# Ans:-
#        Well Posed Learning Problem – A computer program is said to learn from experience E in context to some task T and some performance measure P, if its performance on T, as was measured by P, upgrades with experience E.Any problem can be segregated as well-posed learning problem if it has three traits – Task, Performance and Experience.

# 6. Is machine learning capable of solving all problems? Give a detailed explanation of your answer.
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#        Machine Learning probably can run into all problems for solving but lays down with some concerns which are:
# 
# Ethics: The idea of trusting data and algorithms more than our own judgment has its pros and cons.Obviously, we benefit from these algorithms, otherwise, we wouldn’t be using them in the first place. These algorithms allow us to automate processes by making informed judgments using available data. Sometimes, however, this means replacing someone’s job with an algorithm, which comes with ethical ramifications.
# 
# Deterministic Problem: Machine learning is stochastic, not deterministic. A neural network does not understand Newton’s second law, or that density cannot be negative — there are no physical constraints.
# 
# Lack of Data: Many machine learning algorithms require large amounts of data before they begin to give useful results.

# 7. What are the various methods and technologies for solving machine learning problems? Any two
# of them should be defined in detail.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#        The Various Technologies Used in Machine Learning Problems are: Scikit Learn, Pytorch, Tensorflow, Keras, Python.
# 
# 1. Scikit Learn: Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, Regression, clustering and dimensionality reduction via a consistence interface in Python.
# 2. Tensorflow: It is a open source artificial intelligence library, Using data flow graphs to build models. It allows developers to create large-scale neural networks with many layers. TensorFlow is mainly used for: Classification, Perception, Understanding, Discovering, Prediction and Creation.
#   The Various Methods used in Machine Learning Problems are: Regression, Classification, Clustering, Dimensionality Reductio, Ensemble Methods, Neural Network, Deep Learning, Transfer Learning, Reinforcement Learning, Natural Language Processing, Word Embeddings.
# 
# Regression methods fall within the category of supervised ML. They help to predict or explain a particular numerical value based on a set of prior data, for example predicting the price of a property based on previous pricing data for similar properties.
# 
# Another class of supervised ML, classification methods predict or explain a class value. For example, they can help predict whether or not an online customer will buy a product. The output can be yes or no, buyer or not buyer. But classification methods aren’t limited to two classes. For example, a classification method could help to assess whether a given image contains a car or a truck. In this case, the output will be 3 different values:
# 
# The image contains a car
# The image contains a truck
# The image contains neither a car nor a truck.

# 8. Can you explain the various forms of supervised learning? Explain each one with an example
#     application.
# 
# 
# 
# 
# 
# 
# Ans:-
#      The various forms of supervised learning are explained in detail below:
# 
# 1. Regression: In regression, a single output value is produced using training data.For example, regression can help predict the price of a house based on its locality, size, etc.
# 2. Classification: It involves grouping the data into classes.eg. If you are thinking of extending credit to a person, you can use classification to determine whether or not a person would be a loan defaulter.
# 3. Naive Bayesian Model: The Bayesian model of classification is used for large finite datasets. It is a method of assigning class labels using a direct acyclic graph.
# 4. Decision Trees: A decision tree is a flowchart-like model that contains conditional control statements, comprising decisions and their probable consequences. The output relates to the labelling of unforeseen data.

# 9. What is the difference between supervised and unsupervised learning? With a sample application
#     in each region, explain the differences. 
# 
# 
# 
# 
# 
# 
# Ans:-
#       Supervised earning algorithms are trained using labeled data. Unsupervised learning algorithms are trained using unlabeled data.In unsupervised learning, only input data is provided to the model.
# Examples:
# Supervised Learning : Classification and Regression.
# Unsuperised Learning : Clustering.

# 10. Describe the machine learning process in depth.
# 
# a. Make brief notes on any two of the following:
# 
# MATLAB is one of the most widely used programming languages.
# 
# ii. Deep learning applications in healthcare
# 
# iii. Study of the market basket
# 
# iv. Linear regression (simple)
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#        Make brief notes on any two of the following:
# 1. MATLAB is one of the most widely used programming languages.
# 2. Deep learning applications in healthcare
# 3. Study of the market basket
# 4. Linear regression (simple)

# 11. Make a comparison between:-
# 
# 1. Generalization and abstraction
# 
# 2. Learning that is guided and unsupervised
# 
# 3. Regression and classification
# 
# 
# 
# 
# 
# 
# 
# Ans:-
#       The differences between among the given concepts is:
# 
#     Generalization and abstraction: Abstraction is the process of removing details of objects. And Generalization, then, is the formulation of general concepts from specific instances by abstracting common properties. A concrete object can be looked at as a “subset” of a more generalized object.
# 
#      Learning that is guided and Unsupervised: Supervised learning is the method that trains machines to use data that is well classified and labeled. Whereas Unsupervised learning, on the other hand, is the method that trains machines to use data that is neither classified nor labeled.
# 
#     Regression and classification:
# 
#     Classification Models – Classification models are used for problems where the output variable can be categorized, such as Yes or No, or Pass or Fail. Classification Models are used to predict the category of the data. Real-life examples include spam detection, sentiment analysis, scorecard prediction of exams, etc.
# 
#     Regression Models – Regression models are used for problems where the output variable is a real value such as a unique number, dollars, salary, weight or pressure, for example. It is most often used to predict numerical values based on previous data observations. Some of the more familiar regression algorithms include linear regression, logistic regression, polynomial regression, and ridge regression.
