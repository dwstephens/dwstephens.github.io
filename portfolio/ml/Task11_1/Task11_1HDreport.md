---
title: "Drug Induced Autoimmunity Report"
geometry: "top=2cm, bottom=1.5cm, left=3cm, right=2.5cm"
header-includes: |
   \usepackage{placeins}
   \usepackage{booktabs}
---

# Introduction

Some medicines can have a rare but serious side effect where they trick the body's immune system into attacking itself. This is called Drug-Induced Autoimmunity (DIA). Predicting which drugs might cause this is not easy because it can happen months or even years after a person starts taking a medication, and it depends on many complex factors.

This report is concerned with the results of the research paper "InterDIA: Interpretable prediction of drug-induced autoimmunity through ensemble machine learning approaches" by Huang, Liu and Huang [1]. In their paper, Huang, Liu and Huang present the development of InterDIA, an interpretable machine learning model designed to predict drug-induced autoimmunity (DIA) based on a molecule's physicochemical properties. The InterDIA framework aims to identify potential autoimmune toxicity risks early in the drug development process.

In developing InterDIA, Huang, Liu and Huang tested many machine learning models, focusing on ensemble models. Their final model was based on an Easy Ensemble Classifier (EEC) and achieved an accuracy (ACC) of 85% and an Area Under the Curve (AUC) of 0.8930.

The work presented in this report and the accompanying Jupyter Notebook has two parts:

* **Part 1**: Focuses on reproducing the main results (feature subset RDKit_GA_65) from [1], presented in Table 5 on page 5.

* **Part 2**: Focuses on the design and development of a new machine learning solution for the prediction of DIA.

# Part 1: Reproducing results
This part details the process and results of reproducing the findings presented in Table 5 of [1]. The reproduction adheres strictly to the methods, data, and parameters described by the authors.

## Dataset and features

The dataset used in the paper, [Drug Induced Autoimmunity Prediction](https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction), was obtained from the UC Irvine Machine Learning Repository [2]. The data is licensed under CC BY, allowing it to be freely used for this work. 

The data comprises 192 features, one labelled variable, 'Label' and one categorical variable 'SMILES'. The SMILES (Simplified Molecular Input Line Entry System) "are a standardised text-based language for describing the structure of chemical molecules using short ASCII strings" [3]. For this work, 'Label' is used as the target variable.

The data is already split with an 8:2 ratio into training and external validation sets, provided as separate files. The 8:2 split resulted in 477 compounds for the training set (118 DIA-positive, 359 DIA-negative) and 120 for the external validation set (30 DIA-positive, 90 DIA-negative) representing a 1:3 imbalanced between the positive and negative classes. This exact split was used in this reproduction.

Table 5 of [1] evaluates the performance of five ensemble machine learning models on two optimised feature subsets of different molecular properties ('descriptors'):

  1. RDKit_GA_65: A set of 65 molecular descriptors generated using RDKit and selected via a Genetic Algorithm (GA).
  
  2. RDKit+MOE+DS_RFECV_43: A set of 43 descriptors from RDKit, MOE, and DS platforms, selected using Recursive Feature Elimination with Cross-Validation (RFECV).

This work is only concerned with reproducing the result for the RDKit_GA_65 feature set. The specific list of these 65 features were sourced from the [supplementary data](https://doi.org/10.1016/j.tox.2025.154064) associated with research paper [1]. The supplementary data is provided as an Excel file with the list of features for each subset available in the 'Table S4' sheet.

\FloatBarrier
## Machine learning methods and hyperparameters

Huang, Liu and Huang employed five ensemble machine learning models designed to handle class imbalance (discussed on page 3 of [1]). For their implementation they used Python and various machine learning libraries. The specific version of these libraries were provided in the paper and a listed in Table \ref{tab:lib-versions}. For this work the same versions of the libraries were installed along with Python 3.10.18.

\begin{table}[htb]
\centering
\caption{Machine learning libraries and versions.}
\label{tab:lib-versions}
\begin{tabular}{l|l}
\toprule
\textbf{Library} & \textbf{Version} \\ \midrule
imbalanced-learn & 0.12.3           \\ 
lightgbm         & 3.3.5            \\ 
scikit-learn     & 1.5.1            \\ 
xgboost          & 1.6.1            \\ \bottomrule
\end{tabular}
\end{table}

The tuned hyperparameters for each model are provided in Table S5 of the supplementary material. When conducting this work, it was discovered that not all the required hyperparameters needed to reproduce the work were reported in Table S5. Those missing were 'scale_pos_weight' for the BBC+XGBoost model and the 'random_state', which was required by many of the models. The 'scale_pos_weight' parameter relates to class imbalance and could be deduced from the XGBoost documentation. The 'random_state', on the other hand, could not be inferred and would have required running the models with different values and evaluating the results until the correct random state is found. Fortunately, the authors provided a link in the paper to a GitHub repository [4] where the random state value used for the paper was found. Table \ref{tab:hyper} shows all of the hyperparameters used to reproduce the results of [1].


\begin{table}[htb]
\centering
\caption{Models and their key hyperparameters}
\label{tab:hyper}
\begin{tabular}{l|l|l}
\toprule
\textbf{Model Name}                                                       & \textbf{Base Learner}                                               & \textbf{Key Hyperparameters}                                                                                                                                                                                                                                                                                                 \\ \hline
\begin{tabular}[c]{@{}l@{}}BRF\\ (Balanced Random Forest)\end{tabular}    & Decision Tree                                                       & \begin{tabular}[c]{@{}l@{}}n\_estimators=154, \\ criterion=`gini',\\ max\_depth=15,\\ max\_features=48,\\ random\_state=1\end{tabular}                                                                                                                                                                                       \\ \hline
\begin{tabular}[c]{@{}l@{}}EEC\\ (Easy Ensemble Classifier)\end{tabular}  & \begin{tabular}[c]{@{}l@{}}AdaBoost / \\ Decision Tree\end{tabular} & \begin{tabular}[c]{@{}l@{}}AdaBoost:\\ \quad n\_estimators=178,\\ \quad learning\_rate=0.92,\\ \quad algorithm=`SAMME.R',\\ DecisionTree:\\ \quad max\_depth=7\\ random\_state=1\end{tabular}                                                                    \\ \hline
\begin{tabular}[c]{@{}l@{}}BBC+XGBoost\\ (Balanced Bagging)\end{tabular}  & XGBoost                                                             & \begin{tabular}[c]{@{}l@{}}n\_estimators=172,\\ learning\_rate=0.73,\\ booster=`dart',\\ colsample\_bytree=0.3,\\ colsample\_bynode=1.0,\\ gamma=0.036296772856035525,\\ reg\_lambda=0.06781903189364931,\\ min\_child\_weight=1.0,\\ max\_depth=18,\\ subsample=0.9,\\ scale\_pos\_weight=3,\\ random\_state=1\end{tabular} \\ \hline
\begin{tabular}[c]{@{}l@{}}BBC+GBDT\\ (Balanced Bagging)\end{tabular}     & Gradient Boosting                                                   & \begin{tabular}[c]{@{}l@{}}n\_estimators=107,\\ learning\_rate=0.24,\\ criterion=`friedman\_mse',\\ max\_depth=5,\\ max\_features=4,\\ subsample=0.99,\\ random\_state=1\end{tabular}                                                                                                                                        \\ \hline
\begin{tabular}[c]{@{}l@{}}BBC+LightGBM\\ (Balanced Bagging)\end{tabular} & LightGBM                                                            & \begin{tabular}[c]{@{}l@{}}boosting\_type='gbdt',\\ n\_estimators=112,\\ learning\_rate=0.83,\\ max\_depth=14,\\ num\_leaves=85,\\ colsample\_bytree=0.55,\\ subsample=0.83,\\ reg\_alpha=0.011600450241817575,\\ reg\_lambda=0.12670847895140583,\\ min\_child\_samples=2,\\ random\_state=1\end{tabular}                   \\ \bottomrule
\end{tabular}
\end{table}


\FloatBarrier
## Experimental protocol

The experiment was designed to replicate the workflow of [1] as precisely as possible.

1. Data Loading: The training and validation sets were loaded from the files obtained from the UC Irvine Machine Learning Repository [2]. The feature set list obtained by loading the supplementary material file.

```python
# Load feature lists
# Excel file with supplementary information
file_path = "data/1-s2.0-S0300483X25000204-mmc1.xlsx"

# Load the specific sheet containing the feature lists
supp_df = pd.read_excel(file_path,  sheet_name="Table S4", index_col=None, header=2)

# Extract the RDKit_GA_65 column into a list
rdkit_ga_65 = supp_df.iloc[:, 1].dropna().tolist()

# Print or use the lists
print("RDKit_GA_65")
print(f"Number of descriptors: {len(rdkit_ga_65)}")

feature_sets = {"RDKit_GA_65": rdkit_ga_65,}

# Load datasets
train_df = pd.read_csv("data/DIA_trainingset_RDKit_descriptors.csv")  
test_df = pd.read_csv("data/DIA_testset_RDKit_descriptors.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")
```

2. Feature Selection: The datasets were filtered to retain only the features specified in the RDKit_GA_65 list.  
```python
    # Prepare data for the current feature set
    X_train = train_df[feature_list]  
    y_train = train_df['Label']  
    X_test = test_df[feature_list]  
    y_test = test_df['Label']
```
3. Preprocessing: As described in [1], Z-score normalisation was applied using the StandardScaler from scikit-learn. The StandardScaler was fitted only on the training data to learn the scaling parameters. This fitted scaler was then used to transform both the training and the external validation sets, preventing data leakage.
```python
    # Preprocessing: Z-score normalisation, page 3.
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)
```
4. Model Fitting and Performance Evaluation: For each of the fives models, perform the following:

* A 10-fold Cross-validation using the training data. The performance on the training set was evaluated using out-of-fold predictions from a 10-fold cross-validation process. The authors don't mention using stratification in cross-validation process. This provides a robust estimate of the model's performance on unseen data during training.

```python
        # 10-fold Cross-Validation on training set
        # Paper doesn't mention stratifying the cross-validation, so I won't stratify
        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
          
        # Get out-of-fold predictions and probabilities for the minority class
        y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv, n_jobs=-1)  
        y_proba_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv,
                                       method="predict_proba", n_jobs=-1)[:, 1]

        # Calculate the evaluation metrics for the cross-validation results
        cv_metrics = calculate_metrics(y_train, y_pred_cv, y_proba_cv)
```

* Performance evaluation using the external validation data. Each model was trained on the entire preprocessed training set. The trained model was then used to make predictions on the completely separate external validation set. 
```python
        # External Validation 
        # Train on the full training set
        model.fit(X_train_scaled, y_train)  
          
        # Predict on the external validation set  
        y_pred_ext = model.predict(X_test_scaled)  
        y_proba_ext = model.predict_proba(X_test_scaled)[:, 1]  

        # Calculate the evaluation metrics for the external validation set
        ext_metrics = calculate_metrics(y_test, y_pred_ext, y_proba_ext)  
```
* Calculate and store the false positive and true positive rates for use in Part 2.
```python
        # Compute FPR, TPR, and thresholds
        fpr, tpr, _ = roc_curve(y_test, y_proba_ext)
        roc_auc = auc(fpr, tpr)

        # Store for plotting later
        roc_data[fset_name][model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
```   
  
The metrics used for performance evaluation are the five metrics defined in [1]:

* Area Under the ROC Curve (AUC),
* Accuracy (ACC)
\begin{equation}
    ACC = \frac{TP + TN}{TP + TN + FN + FP}
\end{equation}
* Sensitivity (SEN)
\begin{equation}
    SEN = \frac{TP}{TP + FN}
\end{equation}
* Specificity (SPE)
\begin{equation}
    SPE = \frac{TN}{TN + FP}
\end{equation}
* Matthews Correlation Coefficient (MCC)
\begin{equation}
    MCC= \frac{TP \times TN - FP\times FN}{\sqrt{(FP+TN)(FP+TP)(FN+TN)(FN+TP)}}
\end{equation}
  
where TP is the true positive, TN is the true negative, FP is the false positive and FN is the false negative counts. 

 
\FloatBarrier
## Results and Variations

Table \ref{tab:results} shows the reproduced results from this work alongside the original values from Table 5 for direct comparison.

\begin{table}[htb]
\centering
\caption{Reproduced and original [1] results for RDKit\_GA\_65 feature subset.}
\label{tab:results}
\begin{tabular}{llllll}
\toprule
Model & Evaluation & Metric & Original[1] & Reproduced & Variation \\
\hline
BRF & 10-fold cv & AUC & 0.8764 & 0.8764 & 0.0 \\
 &  & ACC(\%) & 81.76 & 81.76 & 0.0 \\
 &  & SEN(\%) & 83.05 & 83.05 & 0.0 \\
 &  & SPE(\%) & 81.34 & 81.34 & 0.0 \\
 &  & MCC & 0.5841 & 0.5841 & 0.0 \\ 
 & External set & AUC & 0.8878 & 0.8887 & -0.0009 \\
 &  & ACC(\%) & 80.83 & 80.83 & 0.0 \\
 &  & SEN(\%) & 76.67 & 76.67 & 0.0 \\
 &  & SPE(\%) & 82.22 & 82.22 & 0.0 \\
 &  & MCC & 0.5444 & 0.5444 & 0.0 \\ \hline
EEC & 10-fold cv & AUC & 0.8836 & 0.8836 & 0.0 \\
 &  & ACC(\%) & 82.81 & 82.81 & 0.0 \\
 &  & SEN(\%) & 82.2 & 82.2 & 0.0 \\
 &  & SPE(\%) & 83.01 & 83.01 & 0.0 \\
 &  & MCC & 0.5978 & 0.5978 & 0.0 \\
 & External set & AUC & 0.893 & 0.893 & 0.0 \\
 &  & ACC(\%) & 85.0 & 85.0 & 0.0 \\
 &  & SEN(\%) & 83.33 & 83.33 & 0.0 \\
 &  & SPE(\%) & 85.56 & 85.56 & 0.0 \\
 &  & MCC & 0.6413 & 0.6413 & 0.0 \\ \hline
BBC+XGBoost & 10-fold cv & AUC & 0.8499 & 0.8499 & 0.0 \\
 &  & ACC(\%) & 79.87 & 79.87 & 0.0 \\
 &  & SEN(\%) & 77.97 & 77.97 & 0.0 \\
 &  & SPE(\%) & 80.5 & 80.5 & 0.0 \\
 &  & MCC & 0.5327 & 0.5327 & 0.0 \\
 & External set & AUC & 0.8763 & 0.8763 & 0.0 \\
 &  & ACC(\%) & 80.83 & 80.83 & 0.0 \\
 &  & SEN(\%) & 73.33 & 73.33 & 0.0 \\
 &  & SPE(\%) & 83.33 & 83.33 & 0.0 \\
 &  & MCC & 0.5313 & 0.5313 & 0.0 \\ \hline
BBC+GBDT & 10-fold cv & AUC & 0.8773 & 0.8773 & 0.0 \\
 &  & ACC(\%) & 83.65 & 83.65 & 0.0 \\
 &  & SEN(\%) & 74.58 & 74.58 & 0.0 \\
 &  & SPE(\%) & 86.63 & 86.63 & 0.0 \\
 &  & MCC & 0.585 & 0.585 & 0.0 \\
 & External set & AUC & 0.8974 & 0.8974 & 0.0 \\
 &  & ACC(\%) & 85.0 & 85.0 & 0.0 \\
 &  & SEN(\%) & 73.33 & 73.33 & 0.0 \\
 &  & SPE(\%) & 88.89 & 88.89 & 0.0 \\
 &  & MCC & 0.6093 & 0.6093 & 0.0 \\ \hline
BBC+LightGBM & 10-fold cv & AUC & 0.8653 & 0.8653 & 0.0 \\
 &  & ACC(\%) & 82.81 & 82.81 & 0.0 \\
 &  & SEN(\%) & 74.58 & 74.58 & 0.0 \\
 &  & SPE(\%) & 85.52 & 85.52 & 0.0 \\
 &  & MCC & 0.5694 & 0.5694 & 0.0 \\
 & External set & AUC & 0.8915 & 0.8915 & 0.0 \\
 &  & ACC(\%) & 84.17 & 84.17 & 0.0 \\
 &  & SEN(\%) & 73.33 & 73.33 & 0.0 \\
 &  & SPE(\%) & 87.78 & 87.78 & 0.0 \\
 &  & MCC & 0.5926 & 0.5926 & 0.0 \\
\bottomrule
\end{tabular}
\end{table}


\FloatBarrier
### Discussion of Variations
Inspection of Table \ref{tab:results} shows that only the AUC value for the BRF model on the external training set varies from the original published values. A comparison of this value from the paper, 0.8878, with that calculated here, 0.8887, suggests that potentially this is a recording error in the paper, with the last two digits being swapped. It would be very unlikely that the BRF model of this work could match exactly the cross-validation and four out of the five metrics for the external validation set and be a different model to that used to create the original results. Apart from this minor discrepancy the reproduction effort was able to match the original results exactly. This high degree of reproducibility is attributed to the following factors:

* The complete dataset, including the exact train-test split was available.

* The list of features in each feature subset was made available.

* The use of common and well-documented Python libraries (scikit-learn, imbalanced-learn, xgboost, lightgbm) ensured that the underlying algorithms behaved as expected and the exact version numbers used was recorded.

* The modified hyperparameters for each model were record and available.

* Crucially, the random state parameter for all stochastic models and processes, which eliminates variability between runs was able to be located. The authors should have recorded this parameter either in the paper or the supplementary material.

The successful replication in this work confirms that the results reported in the paper are robust, reliable, and were generated through a sound and reproducible experimental protocol.


\FloatBarrier
# Part 2: 
This part details the design and development of a different machine learning solution for the predication of DIA. Whilst [1] focusses on the predication and interpretability of DIA, this work focusses solely on the prediction.

## Proposed Solution Motivation 
The original study by Huang, Liu and Huang [1] successfully developed a high-performance prediction framework, 'InterDIA', by combining multi-strategy feature selection on curated molecular descriptors with advanced ensemble resampling techniques. While effective, this approach relies on generating extensive sets of high-level, pre-defined descriptors and using computationally intensive methods like Genetic Algorithms for feature selection.

The motivation for this proposed solution is to explore an alternative approach that offers a different set of trade-offs between feature representation, model complexity, and interpretability. The core motivations are:

### Fundamental feature representation.
To move from high-level, engineered descriptors to a more fundamental and universal representation of molecular structure using Morgan fingerprints. Fingerprints reduce reliance on specific descriptor calculation software and capture structural information in a more direct, data-driven manner. Morgan fingerprints, also known as Extended Connectivity Fingerprints (ECFPs), are a method of representing a chemical structure as a series of binary digits (a bit vector) [5], [6]. This process involves systematically identifying all circular substructures around each atom within a defined radius [6] or they can be constructed using SMILES [7], [8], [9].

Morgan fingerprints are frequently used as features for training machine learning models in chemoinformatics and drug discovery due to their ability to capture detailed structural information [5], [6], [7]. Some examples of how they have been used in machine learning models include:

* **Neurotoxicity Prediction**: Morgan fingerprints were used in combination with machine learning methods like Random Forest and Support Vector Machine (SVM) for the prediction of the neurotoxicity of chemical compounds in a study [7]. MorganFP-SVM model had an accuracy of 86.56% [7].

* **Prediction of Breast Cancer Cell Inhibition**:
Researchers made predictive models for inhibiting breast cancer cells with Morgan fingerprints as a key molecular representation [5]. Random Forest models with Morgan fingerprints were more efficient compared to other machine learning models in these studies[5].

* **Drug-Induced Liver Toxicity Prediction**: To predict drug-induced liver toxicity, researchers used Morgan fingerprints to calculate the Tanimoto similarity index, which helps to estimate the chemical diversity of a dataset [10].

* **Drug Response Prediction**: Morgan fingerprints were integrated with genetic profiles in deep learning models for drug response prediction [6]. For instance, the addition of 2048-bit Morgan fingerprints to the HiDRA model enhanced drug response prediction accuracy significantly [6].

* **Drug-Drug Interaction Prediction**: To predict drug-drug interactions, a computational method was developed where drugs were represented by their fingerprint features[8].

### Dimensionality reduction.
Systematic dimensionality reduction is applied to handle the high dimensionality of molecular fingerprints, rather than through subset selection, using Principal Component Analysis (PCA). PCA can find hidden variables (principal components) that characterise the most significant axes of structural variance in the entire dataset, perhaps uncovering complex relationships which individual descriptors cannot. PCA has been applied to Morgan fingerprints to reduce the high dimensionality of fingerprint data before it is fed into machine learning models for classification tasks, such as predicting chemical toxicity [7], [10].

### Fundamentally different machine learning method
Huang, Liu and Huang [1] focussed on ensemble methods. In this work Support Vector Machine (SVM) is utilised as the classifier.  While ensemble methods aggregate the predictions of multiple base learners to improve robustness and accuracy, SVM is a single, margin-based classifier that seeks the optimal separating hyperplane. This fundamental difference means SVM does not benefit from the variance reduction or bias correction that ensembles provide, but instead relies on its ability to find a strong, singular decision boundary. SVM has been used to build predictive classification models for various toxicological and pharmaceutical problems [5], [7], [8], [10], [11]. In some cases, the combination of Morgan fingerprints and SVM was found to be a top-performing model [7], [10].

### Alternative imbalance handling
For addressing the significant class imbalance at the algorithmic level with cost-sensitive learning in a
Support Vector Machine (SVM) [12]. Instead of resampling the data (which can either discard majority class information or create synthetic minority samples), this approach modifies the model's learning process to penalise misclassifications of the rare DIA-positive class more heavily.

This 'FP+PCA+SVM' (Fingerprint+PCA+SVM) approach aims to achieve robust predictive performance through a fundamentally different, yet powerful, machine learning methodology.


## Difference with InterDIA
The proposed FP+PCA+SVM solution is substantially different from the InterDIA framework models across every major stage of the machine learning pipeline as described in Table \ref{tab:diff}.

\begin{table}[htb]
\centering
\caption{Comparsion of InterDIA with FP+PCA+SVM}
\label{tab:diff}
\begin{tabular}{lll}
\toprule
\textbf{Pipeline Stage}                                          & \textbf{InterDIA (Best Model)}                                                                                                                                  & \textbf{\begin{tabular}[c]{@{}l@{}}FP+PCA+SVM\\ (Proposed Solution)\end{tabular}}                                                                                 \\ \midrule
\begin{tabular}[c]{@{}l@{}}Feature\\ Representation\end{tabular} & \begin{tabular}[c]{@{}l@{}}High-level, curated molecular\\ descriptors (RDKit, MOE, DS)\end{tabular}                                                            & \begin{tabular}[c]{@{}l@{}}Low-level, structural Morgan\\ fingerprints\end{tabular}                                                                               \\ \hline
\begin{tabular}[c]{@{}l@{}}Feature\\ Engineering\end{tabular}    & \begin{tabular}[c]{@{}l@{}}A small subset of the most \\ relevant descriptors was chosen\\ using methods like Genetic\\ Algorithms (GA) and RFECV.\end{tabular} & \begin{tabular}[c]{@{}l@{}}The entire high-dimensional \\ fingerprint space is transformed\\ and reduced using Principal\\ Component Analysis (PCA).\end{tabular} \\ \hline
\begin{tabular}[c]{@{}l@{}}Imbalance\\ Handling\end{tabular}     & \begin{tabular}[c]{@{}l@{}}Used Easy Ensemble Classifier\\ (EEC) to create balanced data\\ subsets for training an ensemble\\ of models.\end{tabular}           & \begin{tabular}[c]{@{}l@{}}Employs cost sensitive learning\\ within the SVM to adjust the loss\\ function directly.\end{tabular}                                  \\ \hline
Core ML Algorithm                                                & Ensemble of AdaBoost learners.                                                                                                                                  & \begin{tabular}[c]{@{}l@{}}A single, Support Vector Machine\\ (SVM) with a non-linear (RBF)\\ kernel.\end{tabular}                                                \\ \hline
\begin{tabular}[c]{@{}l@{}}Hyperparameter\\ Tuning\end{tabular}  & \begin{tabular}[c]{@{}l@{}}Bayesian Optimization to find\\ optimal parameters.\end{tabular}                                                                     & \begin{tabular}[c]{@{}l@{}}Exhaustive Grid Search with\\ Cross-Validation to find the best\\ combination of PCA and SVM\\ parameters.\end{tabular}                \\ \bottomrule
\end{tabular}
\end{table}


\FloatBarrier
### Model description

The proposed model is a cohesive Python (3.10.18) pipeline constructed using scikit-learn (version 1.5.1). Any reader can implement this model using the details below.

1. **Feature generation: Morgan Fingerprints**  
   * **Library**: RDKit (rdkit.Chem) version 2025.03.6  
   * **Function**: GetMorganGenerator  
   * **Parameters**:
     * radius=2: Captures structural features up to a diameter of 4 bonds.
     * nBits=1028: The fingerprint is a 1024-bit vector, providing a rich but high-dimensional feature space.  
2. **Dimensionality reduction: PCA**  
   * **Library**: sklearn.decomposition.PCA  
   * **Purpose**: To reduce the 1024-dimension fingerprint vector into a smaller, more manageable set of principal components that explain most of the variance in the data.  
   * **Parameter**:  
     * n_components: This is treated as a hyperparameter and tuned during the grid search. It determines the final number of dimensions used by the SVM.  
3. **Classifier: Support Vector Machine (SVM)**  
   * **Library**: sklearn.svm.SVC  
   * **Parameters**:  
     * kernel='rbf': The Radial Basis Function kernel is used to handle complex, non-linear relationships between features and the target.  
     * probability=True: Required to calculate AUC scores.  
     * class_weight='balanced': This is the key to handling data imbalance. It automatically adjusts weights inversely proportional to class frequencies in the input data.  
     * random_state=1:  Parameter for all stochastic models and processes.  
     * C: The regularisation parameter. A smaller C creates a wider margin but allows more misclassifications, while a larger C aims for fewer misclassifications at the risk of over-fitting. This is a tuned hyperparameter.  
     * gamma: The kernel coefficient for 'rbf'. It defines how much influence a single training example has. This is a tuned hyperparameter.  
4. **Hyperparameter tuning: Grid Search CV**  
   * **Library**: sklearn.model\_selection.GridSearchCV  
   * **Process**: The PCA and SVM are combined into a single Pipeline object. GridSearchCV systematically works through all combinations of the parameters defined in the grid below, using 10-fold cross-validation to evaluate each combination.  
   * **Parameter grid**:  
     * n_components: [0.6, 0.7, 0.8]. Fraction values represent the fraction of variance to capture.  
     * C: [0.1, 0.5]. Force the model to be simpler by limiting the max value of C.  
     * gamma: [1, 0.1, 0.01, 0.001].  

To be consistent with [1] the combination yielding the highest mean cross-validated Mathews Correlation Coefficient (MCC) score is selected as the best model. The best parameters were n_components=0.7, C=0.5, and gamma=0.1.


## Experimental protocol

To ensure a fair and direct comparison with [1], the experimental protocol was kept as consistent as possible.

1. Data Loading: The training and validation sets were loaded from the files obtained from the UC Irvine Machine Learning Repository [2]. Only the SMILES feature and target labels are required from the loaded data.

2. Feature Engineering: The SMILES strings in the training and external validation set were converted into 1024-bit Morgan fingerprints. Morgan finger prints are binary and do not need any scaling. 

3. Hyperparameter Tuning: The GridSearchCV object, containing the PCA+SVM pipeline and the parameter grid, was trained on the entire training set of fingerprints and labels. The internal 10-fold cross-validation of the grid search ensures that hyperparameter tuning is robust and does not leak information from the validation set.

4. Performance Evaluation on Training Set: A stratified 10-fold cross-validation was used to incrementally train the model on larger portions of the training data to generate a plot of the learning curve to diagnose the model's performance.

5. Probability Threshold Moving. For some classification problems that have a class imbalance, the default threshold can result in poor performance [13]. A simple approach to improving the performance of a classier that predicts probabilities on an imbalanced classification problem is to tune the threshold used to map probabilities to class labels. Using a stratified 10-fold cross-validation on the training data, out-of-sample probability predictions are used to determine the optimal threshold as the point where the sensitivity and specificity curves intersect. The threshold at the intersection point is returned for use in the evaluation stage.

6. Performance Evaluation on External Validation Set. The model was trained on the entire preprocessed training set. The trained model was then used to make predictions on the completely separate external validation set. 

7. Model Stability Check. To reveal the models' consistency, a subsampling stability test was performed. A loop for 50 iterations is run. In each iteration, the model is trained on a subset (80%) of the complete training data and the model performance evaluated on the external validation set. After all iterations are completed, the performance metrics are aggregated, and interquartile ranges (IQRs) for each metric is visualised. 


## Evaluation Metrics

The metric used for performance evaluation are the sames as shown in Part 1 and [1] being:

* Area Under the ROC Curve (AUC),
* Accuracy (ACC)
* Sensitivity (SEN)
* Specificity (SPE)
* Matthews Correlation Coefficient (MCC)

For model comparison, the primary evaluation metrics used are AUC and ACC.


## Results and Discussion

The performance of the proposed FP+PCA+SVM model was evaluated on the external validation set and compared against the models from [1] on the RDKit_GA_65 feature set, see Table \ref{tab:compare-results}.

\begin{table}[htb]
\centering
\caption{Performance comparison on the external validation set}
\label{tab:compare-results}
\begin{tabular}{llllll}
\toprule
Model & AUC & ACC(\%) & SEN(\%) & SPE(\%) & MCC \\
\midrule
FP+PCA+SVM &    \textbf{0.9185} & 82.50 & \textbf{83.33} & 82.22 &    0.5985 \\
BRF & 0.8878 & 80.83 & 76.67 & 82.22 & 0.5444 \\
EEC & 0.893 & \textbf{85.0} & \textbf{83.33} & 85.56 & \textbf{0.6413} \\
BBC+XGBoost & 0.8763 & 80.83 & 73.33 & 83.33 & 0.5313 \\
BBC+GBDT & 0.8974 & \textbf{85.0} & 73.33 & \textbf{88.89} & 0.6093 \\
BBC+LightGBM & 0.8915 & 84.17 & 73.33 & 87.78 & 0.5926 \\
\bottomrule
\end{tabular}
\end{table}

Figure \ref{fig:compare} shows a comparison of the primary evaluation metrics, AUC and ACC, for FP+PCA+SVM against the five models from [1] discussed in Part 1. The FP+PCA+SVM model outperforms the best model (EEC) from [1] for the AUC, with a score of 0.9185 and is just slightly lower than EEC in accuracy (ACC) with a score of 82.5%. This is an expected and encouraging outcome. The EEC model benefits from a sophisticated, multi-stage feature selection process using domain-specific descriptors, which were meticulously optimised for this exact problem. In contrast, the FP+PCA+SVM model utilises a more generalised, data-driven approach without relying on specialised prior knowledge.

![Comparison of AUC and ACC for this work and models from [1]. \label{fig:compare}](images/model_comparison_subplots.png)

Figure \ref{fig:roc} shows the ROC curves for each model in Part 1 combined with the FP+PCA+SVM model to allow visual comparison of the discriminative power of the models on the 120 compounds in the external validation set.

![ROC Curves on the external validation set for this work and models from [1]. \label{fig:roc}](images/roc_curves.png)

Table \ref{tab:compare-others} displays the performance of four machine learning algorithms employed in the prediction of DIA. FP+PCA+SVM and EEC [1] models are significantly better, while the remaining two models are not good. The new FP+PCA+SVM model possesses the highest AUC (0.9185), indicating it is the best model for distinguishing between the positive and negative classes across all classification thresholds. It also possesses the highest sensitivity (83.33%) in common with the EEC model, i.e., it is efficient in classifying actual positive examples accurately, which results from the shifting of the probability threshold. 

The EEC model demonstrates the best, most well-balanced, and strongest performance overall, as observed from its highest MCC (0.6413). The model also possesses high accuracy and the second-highest specificity, which means it is highly powerful in its prediction.

The CatBoost [14] model presents a classic case of misleading accuracy. While its accuracy is the highest (90.24%), this is driven by an extremely high specificity (97.22%) and an inferior sensitivity (40.00%). This severe imbalance indicates that the model excels at identifying negative cases but struggles to identify more than half of the positive cases, rendering it unreliable for practical use.

Finally, the MACCS_SVM [11] model shows less effective performance across the board. All of its metrics are significantly lower than the top two models, with a particularly low MCC of 0.33, indicating weaker predictive power.

\begin{table}[htb]
\centering
\caption{}
\label{tab:compare-others}
\begin{tabular}{llllll}
\toprule
Model                  & AUC    & ACC(\%) & SEN(\%) & SPE(\%) & MCC    \\ \midrule
FP+PCA+SVM (this work) & 0.9185 & 82.50   & 83.33   & 82.22   & 0.5985 \\
EEC {[}1{]}            & 0.893  & 85.0    & 83.33   & 85.56   & 0.6413 \\
MACCS\_SVM {[}x{]}     & 0.84   & 76.26   & 75.76   & 76.31   & 0.33   \\
CatBoost {[}y{]}       & 0.7    & 90.24   & 40.00   & 97.22   & 0.47  \\
\bottomrule
\end{tabular}
\end{table}

**Strengths of the proposed model**:  

* Simplicity and Generality: The FP+PCA+SVM model is conceptually simpler and does not rely on pre-curated, high-level descriptor sets. It can be readily applied to any chemical dataset with only SMILES strings as input.  

* Efficient Imbalance Handling: By handling class imbalance at the algorithm level, it avoids the complexities and potential information loss associated with data resampling techniques. 
‌
* Alternative Views: While PCA components are not as directly interpretable as descriptors like 'lipophilicity', they explain global structural differences most responsible for class discrimination and offer an alternative and perhaps useful view toward the design of drugs.

**Weaknesses of the proposed model**:

* Lack of Interpretability: This is the most significant weakness in drug design. The goal of modelling is often not just to predict but to understand. The use of Morgan fingerprints and PCA abstract away the chemical structure of the compounds, resulting in a model that might be predictive, but it cannot tell chemists why it thinks a molecule is active. It fails to provide actionable insights for designing better compounds.

* Scalability with Large Datasets: The training time for SVMs can scale poorly with the number of compounds. For very large datasets (e.g., screening millions of molecules), training an SVM can be much slower compared to other algorithms like tree-based models.


## Conclusion

This work successfully demonstrates that a substantially different machine learning pipeline can also effectively model the complex structure-toxicity relationship of DIA. The FP+PCA+SVM serves as a powerful and simpler alternative to the InterDIA framework for predicting DIA. Further investigation is required to determine if the model's predictive components can be mapped back to chemically intuitive structural features. 


\FloatBarrier
# References

[1] L. Huang, P. Liu, and X. Huang, “InterDIA: Interpretable prediction of drug-induced autoimmunity through ensemble machine learning approaches,” Toxicology, vol. 511, p. 154064, Feb. 2025, doi: https://doi.org/10.1016/j.tox.2025.154064.


[2] Xiaojie Huang, "Drug-induced Autoimmunity Prediction," UCI Machine Learning Repository, 2024. doi: 10.24432/C5168C.


[3] “SMILES,” DBpedia, 2025. https://dbpedia.org/ontology/smiles (accessed Sep. 20, 2025).


[4] Xiaojie Huang, "InterDIA," Feb. 2024. [Source code]. Available: https://github.com/Huangxiaojie2024/InterDIA. [Accessed: Sep. 20, 2025].
‌

[5] S. He et al., “Machine Learning Enables Accurate and Rapid Prediction of Active Molecules Against Breast Cancer Cells,” Frontiers in Pharmacology, vol. 12, Dec. 2021, doi: https://doi.org/10.3389/fphar.2021.796534.


‌[6] M. Xiao et al., “Drug molecular representations for drug response predictions: a comprehensive investigation via machine learning methods,” Scientific Reports, vol. 15, no. 1, Jan. 2025, doi: https://doi.org/10.1038/s41598-024-84711-7.


[7] Y. Gao, J. Mu, K. Liu, and M. Wang, “Integrating molecular fingerprints with machine learning for accurate neurotoxicity prediction: an observational study,” Advanced technology in neuroscience ., vol. 2, no. 3, pp. 109–115, May 2025, doi: https://doi.org/10.4103/atn.atn-d-24-00034.

‌
[8] B. Ran, L. Chen, M. Li, Y. Han, and Q. Dai, “Drug-Drug Interactions Prediction Using Fingerprint Only,” Computational and Mathematical Methods in Medicine, vol. 2022, pp. 1–14, May 2022, doi: https://doi.org/10.1155/2022/7818480.


‌[9] M. Lovrić et al., “Should We Embed in Chemistry? A Comparison of Unsupervised Transfer Learning with PCA, UMAP, and VAE on Molecular Fingerprints,” Pharmaceuticals, vol. 14, no. 8, p. 758, Aug. 2021, doi: https://doi.org/10.3390/ph14080758.


[10] K. Jaganathan, H. Tayara, and K. T. Chong, “Prediction of Drug-Induced Liver Toxicity Using SVM and Optimal Descriptor Sets,” International Journal of Molecular Sciences, vol. 22, no. 15, p. 8073, Jul. 2021, doi: https://doi.org/10.3390/ijms22158073.


[11] H. Guo et al., “Modeling and insights into the structural characteristics of drug-induced autoimmune diseases,” Frontiers in Immunology, vol. 13, Oct. 2022, doi: https://doi.org/10.3389/fimmu.2022.1015409.


[12] A. Fernández, S. García, M. Galar, R. C. Prati, B. Krawczyk, and F. Herrera, “Cost-Sensitive Learning,” in Learning from Imbalanced Data Sets, Cham: Springer International Publishing, 2018. doi: https://doi.org/10.1007/978-3-319-98074-4.


[13] J. Brownlee, Imbalanced Classification with Python. Machine Learning Mastery, 2020.


[14] Y. Wu, J. Zhu, P. Fu, W. Tong, H. Hong, and M. Chen, “Machine Learning for Predicting Risk of Drug-Induced Autoimmune Diseases by Structural Alerts and Daily Dose,” International Journal of Environmental Research and Public Health, vol. 18, no. 13, p. 7139, Jul. 2021, doi: https://doi.org/10.3390/ijerph18137139.
