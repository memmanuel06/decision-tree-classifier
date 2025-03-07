This project implements a Decision Tree classifier using the Car Evaluation Dataset. The implementation focuses on entropy, information gain, and recursive tree construction.

Features:
  Custom-built decision tree (no external libraries like Scikit-learn)
  Handles categorical data using encoding
  Computes entropy and information gain for feature selection
  Supports tree depth tuning for overfitting/underfitting analysis
  Generates confusion matrix and accuracy metrics
  Includes pruning mechanism and logging for decision steps

The Car Evaluation Dataset is sourced from the UCI Machine Learning Repository. It contains categorical features describing car conditions with six attributes:
  Buying price
  Maintenance cost
  Number of doors
  Number of persons
  Lug boot size
  Safety

