# Apache Hive Bug Prediction Models

This initiative aims to enhance code quality by predicting bugs before releasing
 new code versions. By identifying files likely to contain bugs, we can prioritize
  testing efforts, especially in the final stages before a release. Here, we leverage 
  machine learning models to predict buggy files, optimizing our testing 
  resources and improving software reliability.

## Requirements
- Git: Clone the Hive repository.
- Python 3.8+: Programming environment.
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- SciTools Understand: For code metrics collection.

## Data Source
- [Jira Bug Collection](https://issues.apache.org/jira/projects/HIVE/issues/HIVE-13282?filter=allopenissues)
- [Apache Hive Software](https://github.com/apache/hive)


## Instructions
To get this project up-and-running, clone this repository (nicknamed
*project repo* trhoughout) and the Apache Hive repository (*hive repo*).

The code is spread out in three notebooks, organized as follows. Execute it sequentially 

### 1. Data Extraction
The data extraction process begins with utilizing a the *Data_Extraction.ipynb* that contains the necessary code to efficiently gather and preprocess bug reports from Jira. The first step involves fetching these bug reports, followed by removing redundant entries and concatenating the data to ensure a streamlined dataset. Once the bug reports are curated, the next phase identifies the specific Java and C++ files affected by these bugs, providing a clear focus for subsequent analysis. `NEW` The variable `Priority`, representing the sverity of each bug found was included from the data extraction.

### 2. UND Data Collection
Independent variables for each file across different versions of Hive are gathered using *SciTools Understand*, facilitating a comprehensive understanding of the factors influencing bug occurrences.


### 3. Data Cleanup
Following the extraction, the *Data_Cleanup.ipynb* stage ensures that the dataset is refined and ready for analysis. This involves identifying files that contain bugs, which helps in isolating the problematic areas within the codebase. Further refinement is achieved by adding classes and methods to the processed files, enriching the dataset with relevant structural information. This meticulous cleanup process is crucial for maintaining data integrity and enhancing the accuracy of the subsequent modeling and analysis steps.

### 3. `NEW` Additonnal Metrics Collection
The *Data_Additionnal_Metrics.ipynb* is designed to analyze and extract additional metrics related to software development from an Apache Hive repository. It processes a list of versions and corresponding commit hashes, calculates changes (lines added/removed) between versions for each file, and gathers a range of metrics such as the number of commits affecting a file, bug fix commits, developer involvement, developer expertise, time between commits, and changes to code comments. The script leverages Git to fetch commit data, processes it efficiently, and updates CSV files with these metrics for machine learning training purposes, ensuring computational efficiency by batch-processing files and using concurrent operations.

### 4. `NEW` Data Partition & Trainning
This final notebook `Data_Partition_Trainning.ipynb` facilitates the preparation, training, and evaluation of machine learning models to predict bugs in software files. It begins by loading previously processed data with additional metrics, selecting and cleaning features, handling missing values, and identifying outliers. Correlated features are removed to enhance model interpretability. The data is stratified into training and testing sets, with oversampling and undersampling applied to address class imbalance using SMOTE. Logistic Regression and Random Forest classifiers are trained on the data, with custom class weights to address imbalance. The models are evaluated on various performance metrics, such as ROC-AUC and precision-recall, and their feature importance is visualized through nomograms. Finally, multinomial regression and classification are implemented to predict a categorical target variable (`Priority`) across multiple classes, leveraging similar preprocessing and evaluation techniques.

# Licence
This project is licensed under the MIT License. See the LICENSE file for details.
Contact

# Contact
For questions or suggestions, please contact:

    Name: Nicolas Richard
    Email: nicolas.richard.1997@gmail.com
