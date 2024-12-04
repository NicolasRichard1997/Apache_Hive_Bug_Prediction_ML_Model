#!/usr/bin/env python
# coding: utf-8

# # Additionnal Metrics
# As per the specification of the personnal project, we'll try and gather additionnal metrics to improve our model. Since this notebook is very compute-intensive, it was executed directly in the terminal after converting this notebook to a python file.
# ## 1. Lines added & deleted from a given version
# We'll begin by creating a dictionnary of versions and corresponding commits from the file `Hive_Last_Commits.csv`previously created

# In[34]:


import os
import glob
import re
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
import subprocess
from datetime import datetime
from statistics import mean
import logging
from datetime import datetime
from statistics import mean
import pytz
import git
from typing import Iterable, Dict, Tuple
import math


# In[4]:


project_repo = Path("/home/nicolas-richard/Desktop/.Apache_Hive_Bug_Prediction_ML_Model/")
hive_repo = Path("/home/nicolas-richard/Desktop/.Apache_Hive/")


# In[5]:


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

    output_path = output_directory / f"UND_{df_version}.csv"
    df.to_csv(output_path, index=False)
    print(f"=== Updated file saved as {output_path} ===\n")


# ## 2. Commits

# ### 2.1 Commits Affecting the File in a Given version
# First, we'll fetch the `CommitsAffectingFileInCurrentVersion`, `CommitsFixingBugInFileInCurrentVersion`, `CommitsAffectingFileInPreviousVersions` variables and save them to our additionnal metrics files.

# In[33]:


version_commits_file = "Hive_Last_Commits.csv"

def load_version_commits(version_commits_file):
    """Load version and commit mapping from a CSV file."""
    version_commits = []
    with open(version_commits_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            version_commits.append((row[0].strip(), row[1].strip()))
    return version_commits

def compare_versions(version1, version2):
    """Compare two semantic versions (e.g., 2.1.0 and 2.0.1)."""
    v1 = list(map(int, version1.split('.')))
    v2 = list(map(int, version2.split('.')))
    return v1 <= v2

def collect_metrics(hive_repo, version_commits, target_file_name, df_version):
    """Collect metrics for a target file."""
    repo = git.Repo(hive_repo)
    metrics = defaultdict(dict)
    all_previous_commits = []

    for i, (version, commit_hash) in enumerate(version_commits):
        if not compare_versions(version, df_version):
            continue

        try:
            commits_affecting_file = list(repo.iter_commits(f"{commit_hash}..HEAD", paths=target_file_name))
        except Exception as e:
            print(f"Error fetching commits for {target_file_name}: {e}")
            continue
        
        bug_fix_keywords = ["fix", "bug", "issue", "HIVE-"]
        bug_fix_commits = [
            c for c in commits_affecting_file if any(keyword in c.message.lower() for keyword in bug_fix_keywords)
        ]

        if i > 0:
            try:
                previous_commit_hash = version_commits[i - 1][1]
                previous_commits = list(repo.iter_commits(f"{previous_commit_hash}..{commit_hash}", paths=target_file_name))
                all_previous_commits.extend(previous_commits)
                all_previous_commits = list(set(all_previous_commits))  
            except Exception as e:
                print(f"Error fetching previous commits for {target_file_name}: {e}")
                previous_commits = []

        metrics[version] = {
            "num_commits_in_version": len(commits_affecting_file),
            "num_bug_fix_commits": len(bug_fix_commits),
            "num_commits_in_previous_versions": len(all_previous_commits),
        }
    return metrics

def display_metrics(metrics):
    """Display metrics in a readable format."""
    for version, data in metrics.items():
        print(f"  - Commits affecting file in version: {data['num_commits_in_version']}")
        print(f"  - Bug fix commits in version: {data['num_bug_fix_commits']}")
        print(f"  - Commits in previous versions: {data['num_commits_in_previous_versions']}")
        print()

if __name__ == "__main__":
    version_commits = load_version_commits(version_commits_file)
    files = sorted([
        os.path.join(project_repo, "UND_hive_additional_metrics", f) 
        for f in os.listdir(os.path.join(project_repo, "UND_hive_additional_metrics")) 
    ])

    for file in files:
        df_version = file.split("_")[-1] 
        df = pd.read_csv(file)
        df["CommitsAffectingFileInCurrentVersion"] = 0
        print(f'"CommitsAffectingFileInCurrentVersion" added to version {df_version}')
        df["CommitsFixingBugInFileInCurrentVersion"] = 0
        print(f'"CommitsFixingBugInFileInCurrentVersion" added to version {df_version}')
        df["CommitsAffectingFileInPreviousVersions"] = 0
        print(f'"CommitsAffectingFileInPreviousVersions" added to version {df_version}')


        for index, row in df.iterrows():
            target_file_name = row["FileName"]
            try:
                print(f"\n\nVersion Metrics for {target_file_name} in version <= {df_version}")
                metrics = collect_metrics(hive_repo, version_commits, target_file_name.strip(), df_version)
                display_metrics(metrics)

                df.loc[index, "CommitsAffectingFileInCurrentVersion"] = metrics[df_version]["num_commits_in_version"]
                df.loc[index, "CommitsFixingBugInFileInCurrentVersion"] = metrics[df_version]["num_bug_fix_commits"]
                df.loc[index, "CommitsAffectingFileInPreviousVersions"] = metrics[df_version]["num_commits_in_previous_versions"]
                
            except Exception as e:
                print(f"Error processing {target_file_name}: {e}")

        df.to_csv(file, index=False)
        print(f"=== Updated file saved as {file} ===\n")
    print("\n\n\nCommit version processing successful\n\n\n")

