#!/usr/bin/env python
# coding: utf-8

# # Additionnal Metrics
# As per the specification of the personnal project, we'll try and gather additionnal metrics to improve our model. 
# ## 1. Lines added & deleted from a given version
# We'll begin by creating a dictionnary of versions and corresponding commits from the file `Hive_Last_Commits.csv`previously created

# In[1]:


import os
import glob
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict


# In[2]:


project_repo = Path("/home/nicolas-richard/Desktop/.Apache_Hive_Bug_Prediction_ML_Model/")
hive_repo = Path("/home/nicolas-richard/Desktop/.Apache_Hive/")


# In[3]:


last_commits = open(os.path.join(project_repo, "Hive_Last_Commits.csv"), "r")

versions = []
commits_by_version = {}

for i, line in enumerate(last_commits.readlines()):
    if i == 0:
        continue

    parts = line.strip().split(",")
    version = parts[0]
    commit = parts[1]

    versions.append(version)
    commits_by_version[version] = [commit]  
last_commits.close()

print(commits_by_version)


# In[ ]:


UND_hive_updated_directory = project_repo / "UND_hive_updated_data"
output_directory = project_repo / "UND_hive_additional_metrics"
os.makedirs(output_directory, exist_ok=True)

csv_files = sorted([f for f in os.listdir(UND_hive_updated_directory) if f.endswith('.csv')])

for file in csv_files:
    df = pd.read_csv(UND_hive_updated_directory / file)
    df_version = file.split("_")[1]

    print(f"\n=== Processing version: {df_version} ===")

    for another_file in csv_files:
        another_version = another_file.split("_")[1]
    
        df[f"LinesAddedSince{another_version}"] = 0
        print(f"LinesAddedSince{another_version} added to version {df_version}")
        df[f"LinesRemovedSince{another_version}"] = 0
        print(f"LinesRemovedSince{another_version} added to version {df_version}")


        if another_version >= df_version: 
            continue
            
        print(f"  Comparing with earlier version: {another_version}")
        another_df = pd.read_csv(os.path.join(UND_hive_updated_directory, another_file))

        for index, row in df.iterrows():
            file_name = row["FileName"]
            line_count = row["CountLine"]

            matching_rows = another_df[another_df["FileName"] == file_name]

            if not matching_rows.empty:
                another_line_count = matching_rows.iloc[0]["CountLine"]
                print(f"    - {file_name} found in version {another_version}")

                if line_count > another_line_count:
                    added_lines = line_count - another_line_count
                    df.loc[index, f"LinesAddedSince{another_version}"] = added_lines
                    print(f"      Lines added: {added_lines}")
                elif line_count < another_line_count:
                    removed_lines = another_line_count - line_count
                    df.loc[index, f"LinesRemovedSince{another_version}"] = removed_lines
                    print(f"      Lines removed: {removed_lines}")

    output_path = output_directory / f"UND_{df_version}"
    df.to_csv(output_path, index=False)
    print(f"=== Updated file saved as {output_path} ===\n")


# ## 2. 
