Pokemon Type1 Classification using Decision Tree and Random Forest in R

This project demonstrates a multi-class classification task using R. The goal is to predict the primary type (type1) of a Pokémon based on its attributes using Decision Tree and Random Forest models.

Dataset

Source: https://raw.githubusercontent.com/ccscaiado/MLRepo/refs/heads/main/Assignment%202%20Datasets/Pokemon/pokemon.csv  
Target variable: type1  
Features used: hp, attack, defense, sp_attack, sp_defense, speed, height_m, weight_kg, base_egg_steps, base_happiness, capture_rate, experience_growth, percentage_male, is_legendary, generation  
Features excluded: name, japanese_name, type2, classification, abilities, pokedex_number

Data Preprocessing

Converted variables to appropriate data types (factor, numeric)  
Imputed missing values using the median  
Removed high-cardinality and redundant columns  
Final dataset stored in variable pokemon_2

Exploratory Data Analysis

Summary statistics generated for numeric columns  
Plotted Pokémon count by primary type using ggplot2

Modeling

Decision Tree using rpart  
Used information gain as the split criterion  
Set parameters: minsplit = 10, cp = 0.005, maxdepth = 5  
Applied pruning with cp = 0.02  
Visualized the tree and plotted variable importance  
Evaluated using confusion matrix to calculate accuracy and Kappa

Random Forest using randomForest  
Used 200 trees and mtry = 6  
Generated variable importance plot  
Evaluated using confusion matrix to calculate accuracy and Kappa

Model Evaluation

Compared Decision Tree and Random Forest using these metrics:  
Training Accuracy  
Testing Accuracy  
Training Kappa  
Testing Kappa  
Plotted bar charts to compare performance between the two models

Required Packages

tidyverse  
rpart  
caret  
ggplot2  
randomForest  
pROC  
Install them using install.packages() in R

How to Run

Make sure the file is named Pokemon_type1.R  
Place it in your R working directory  
Run the script with the command: source("Pokemon_type1.R")
