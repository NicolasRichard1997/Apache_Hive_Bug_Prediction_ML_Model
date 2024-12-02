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
The data extraction process begins with utilizing a the *Data_Extraction.ipynb* that contains the necessary code to efficiently gather and preprocess bug reports from Jira. The first step involves fetching these bug reports, followed by removing redundant entries and concatenating the data to ensure a streamlined dataset. Once the bug reports are curated, the next phase identifies the specific Java and C++ files affected by these bugs, providing a clear focus for subsequent analysis. 

### 2. UND Data Collection
Independent variables for each file across different versions of Hive are gathered using *SciTools Understand*, facilitating a comprehensive understanding of the factors influencing bug occurrences.


### 3. Data Cleanup
Following the extraction, the *Data_Cleanup.ipynb* stage ensures that the dataset is refined and ready for analysis. This involves identifying files that contain bugs, which helps in isolating the problematic areas within the codebase. Further refinement is achieved by adding classes and methods to the processed files, enriching the dataset with relevant structural information. This meticulous cleanup process is crucial for maintaining data integrity and enhancing the accuracy of the subsequent modeling and analysis steps.

### 3. Data Partition_Trainning
The final phase, contained in the *Data_Partion_Trainning.ipynb*, encompasses several critical steps to prepare the data for machine learning applications. Initially, data preparation involves selecting the most relevant features and organizing the data appropriately. This is followed by partitioning the data and conducting correlation analysis to understand the relationships between variables. Techniques such as over-sampling and under-sampling are employed to balance the dataset, ensuring robust model training. Two models, Logistic Regression and Random Forest, are then trained and compared on the same dataset to evaluate their performance. Interpretability is addressed through the use of nomograms and the identification of top features for both models. Additionally, the analysis includes a comparison of minor project versions to observe differences in metrics and model behavior. Finally, data is regrouped by major version to identify the most impactful features for each version, providing insights into how different major releases influence bug metrics and model performance.

# Licence
This project is licensed under the MIT License. See the LICENSE file for details.
Contact

# Contact
For questions or suggestions, please contact:

    Name: Nicolas Richard
    Email: nicolas.richard.1997@gmail.com
