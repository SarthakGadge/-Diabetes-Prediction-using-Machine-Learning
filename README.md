<!DOCTYPE html>
<html lang="en">
<body>

<h1>Diabetes Prediction Using Support Vector Machine (SVM)</h1>

<h2>Overview</h2>
<p>This project aims to predict the likelihood of diabetes in patients using a Support Vector Machine (SVM) classifier. The dataset used for training and testing the model is the well-known Pima Indians Diabetes Database. The project involves data preprocessing, feature selection, model training, and evaluation.</p>

<h2>Table of Contents</h2>
<ol>
    <li>Installation</li>
    <li>Dataset</li>
    <li>Data Preprocessing</li>
    <li>Model Training</li>
    <li>Model Evaluation</li>
    <li>Usage</li>
    <li>Results</li>
    
</ol>

<h2 id="installation">Installation</h2>
<p>To run this project, you'll need Python and the following libraries:</p>
<ul>
    <li>numpy</li>
    <li>pandas</li>
    <li>scikit-learn</li>
    <li>matplotlib (optional, for plotting)</li>
</ul>
<p>You can install these packages using pip:</p>
<pre><code>pip install numpy pandas scikit-learn matplotlib</code></pre>

<h2 id="dataset">Dataset</h2>
<p>The dataset used in this project is the Pima Indians Diabetes Database, which is available from the UCI Machine Learning Repository. The dataset contains 768 samples with 8 features each and a binary target variable indicating the presence or absence of diabetes.</p>

<h2 id="data-preprocessing">Data Preprocessing</h2>
<p>Before training the model, the data needs to be preprocessed:</p>
<ol>
    <li><strong>Handling Missing Values:</strong> Check for and handle any missing values in the dataset.</li>
    <li><strong>Feature Scaling:</strong> Normalize or standardize the feature values to ensure they are on a similar scale.</li>
    <li><strong>Train-Test Split:</strong> Split the data into training and testing sets, typically using an 80-20 split.</li>
</ol>

<h2 id="model-training">Model Training</h2>
<p>We use the SVM classifier from scikit-learn to train the model.</p>


<h2 id="model-evaluation">Model Evaluation</h2>
<p>After training, we evaluate the model using the testing set</p>


<h2 id="usage">Usage</h2>
<ol>
    <li><strong>Load the Data:</strong></li>
    <pre><code>
import pandas as pd
data = pd.read_csv('diabetes.csv')
    </code></pre>
    <li><strong>Preprocess the Data:</strong></li>
    <pre><code># Handling missing values, feature scaling, train-test split</code></pre>
    <li><strong>Train the Model:</strong></li>
    <pre><code>
model = SVC(kernel='linear')
model.fit(X_train, y_train)
    </code></pre>
    <li><strong>Evaluate the Model:</strong></li>
    <pre><code>
y_pred = model.predict(X_test)
# Print evaluation metrics
    </code></pre>
</ol>

<h2 id="results">Results</h2>
<p>The results section should include the evaluation metrics obtained from the test set. For instance:</p>
<ul>
    <li>Accuracy: 0.80</li>
    
</ul>
<p>These results indicate the performance of the SVM model in predicting diabetes.</p>

</body>
</html>
