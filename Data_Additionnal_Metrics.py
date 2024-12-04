#!/usr/bin/env python
# coding: utf-8

# # Additionnal Metrics
# As per the specification of the personnal project, we'll try and gather additionnal metrics to improve our model. Since this notebook is very compute-intensive, it was executed directly in the terminal after converting this notebook to a python file.
# ## 1. Lines added & deleted from a given version
# We'll begin by creating a dictionnary of versions and corresponding commits from the file `Hive_Last_Commits.csv`previously created

# In[39]:


import csv
import glob
import logging
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Tuple
import git
import pandas as pd
import pytz


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
# In this section, we'll define functions to generate the metrics for commits affecting the file, developpers having worked on the project and their expertise. We'll begin by defining helper functions before defining metrics collection procedure and executing it all at once in order to speed up the process and minimize read-write operations.

# In[44]:


def extract_version_from_filename(file_name):
    match = re.search(r'(\d+\.\d+\.\d+)', file_name)
    if match:
        return match.group(1)
    raise ValueError(f"Version not found in file name: {file_name}")

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
    """Return True if version1 is less than or equal to version2."""
    v1 = list(map(int, version1.split('.')))
    v2 = list(map(int, version2.split('.')))
    return v1 <= v2

def get_commits(repo, commit_range, target_file_name):
    """Fetch commits affecting the file within a commit range."""
    try:
        commits = list(repo.iter_commits(commit_range, paths=target_file_name))
        return commits
    except Exception as e:
        print(f"Error fetching commits for {target_file_name}: {e}")
        return []


# For the "developper expertise" and "time between commits", we'll define helper functions as well.

# In[45]:


def get_developer_experiences(repo):
    """Get total number of commits made by each developer in the entire project."""
    developer_experiences = defaultdict(int)
    print("Fetching all commits in the repository for developer experiences...")
    for commit in repo.iter_commits():
        developer = commit.author.email
        developer_experiences[developer] += 1
    return developer_experiences

def calculate_expertise_metrics(commits):
    """Calculate expertise metrics from a list of commits."""
    developers = set(commit.author.email for commit in commits)
    num_developers = len(developers)
    expertise = {}
    for developer in developers:
        total_commits = developer_experiences.get(developer, 0)
        expertise[developer] = total_commits
    if expertise:
        avg_expertise = sum(expertise.values()) / len(expertise)
        min_expertise = min(expertise.values())
    else:
        avg_expertise = 0
        min_expertise = 0
    return num_developers, avg_expertise, min_expertise

def calculate_time_metrics(commits):
    """Calculate average time between commits."""
    if len(commits) >= 3:
        commit_dates = sorted([commit.committed_datetime for commit in commits])
        time_diffs = [(commit_dates[i+1] - commit_dates[i]).total_seconds() for i in range(len(commit_dates)-1)]
        avg_time_between_commits = sum(time_diffs) / len(time_diffs)
    else:
        avg_time_between_commits = None
    return avg_time_between_commits


# In[46]:


def collect_metrics(hive_repo, version_commits, target_file_name, df_version):
    """Collect metrics for a target file."""
    global developer_experiences  
    repo = git.Repo(hive_repo)
    metrics = defaultdict(dict)

    relevant_versions = [vc for vc in version_commits if compare_versions(vc[0], df_version)]

    for i, (version, commit_hash) in enumerate(relevant_versions):
        commits_affecting_file = get_commits(repo, f"{commit_hash}..HEAD", target_file_name)

        bug_fix_keywords = ["fix", "bug", "issue", "HIVE-"]
        bug_fix_commits = [
            c for c in commits_affecting_file if any(keyword in c.message.lower() for keyword in bug_fix_keywords)
        ]

        all_previous_commits = []
        if i > 0:
            previous_commit_hash = relevant_versions[i - 1][1]
            previous_commits = get_commits(repo, f"{previous_commit_hash}..{commit_hash}", target_file_name)
            all_previous_commits.extend(previous_commits)
            all_previous_commits = list(set(all_previous_commits)) 

        num_devs_in_version, avg_expertise_in_version, min_expertise_in_version = calculate_expertise_metrics(commits_affecting_file)

        num_devs_in_prev_versions, avg_expertise_in_prev_versions, min_expertise_in_prev_versions = calculate_expertise_metrics(all_previous_commits)

        avg_time_between_commits_in_version = calculate_time_metrics(commits_affecting_file)
        avg_time_between_commits_in_prev_versions = calculate_time_metrics(all_previous_commits)

        metrics[version] = {
            "num_commits_in_version": len(commits_affecting_file),
            "num_bug_fix_commits": len(bug_fix_commits),
            "num_commits_in_previous_versions": len(all_previous_commits),
            "num_developers_in_version": num_devs_in_version,
            "num_developers_in_previous_versions": num_devs_in_prev_versions,
            "avg_expertise_in_version": avg_expertise_in_version,
            "avg_expertise_in_previous_versions": avg_expertise_in_prev_versions,
            "min_expertise_in_version": min_expertise_in_version,
            "min_expertise_in_previous_versions": min_expertise_in_prev_versions,
            "avg_time_between_commits_in_version": avg_time_between_commits_in_version,
            "avg_time_between_commits_in_previous_versions": avg_time_between_commits_in_prev_versions
        }
    return metrics

def display_metrics(metrics):
    """Display metrics in a readable format."""
    for version, data in metrics.items():
        print(f"Version: {version}")
        print(f"  - Commits affecting file in version: {data['num_commits_in_version']}")
        print(f"  - Bug fix commits in version: {data['num_bug_fix_commits']}")
        print(f"  - Commits in previous versions: {data['num_commits_in_previous_versions']}")
        print(f"  - Number of Developers in version: {data['num_developers_in_version']}")
        print(f"  - Number of Developers in previous versions: {data['num_developers_in_previous_versions']}")
        print(f"  - Average Expertise in version: {data['avg_expertise_in_version']}")
        print(f"  - Average Expertise in previous versions: {data['avg_expertise_in_previous_versions']}")
        print(f"  - Minimum Expertise in version: {data['min_expertise_in_version']}")
        print(f"  - Minimum Expertise in previous versions: {data['min_expertise_in_previous_versions']}")
        print(f"  - Average Time Between Commits in version: {data['avg_time_between_commits_in_version']}")
        print(f"  - Average Time Between Commits in previous versions: {data['avg_time_between_commits_in_previous_versions']}")
        print()


# In[47]:


if __name__ == "__main__":
    version_commits_file = "Hive_Last_Commits.csv"
    version_commits = load_version_commits(version_commits_file)
    files = sorted([
        os.path.join(project_repo, "UND_hive_additional_metrics", f) 
        for f in os.listdir(os.path.join(project_repo, "UND_hive_additional_metrics")) 
    ])

    repo = git.Repo(hive_repo)
    developer_experiences = get_developer_experiences(repo)

    for file in files:
        df_version = extract_version_from_filename(file)
        print(f"Extracted version: {df_version}")

        df = pd.read_csv(file)
        df["CommitsAffectingFileInCurrentVersion"] = 0
        df["CommitsFixingBugInFileInCurrentVersion"] = 0
        df["CommitsAffectingFileInPreviousVersions"] = 0
        df["NumDevelopersInVersion"] = 0
        df["NumDevelopersInPreviousVersions"] = 0
        df["AvgExpertiseInVersion"] = 0.0
        df["AvgExpertiseInPreviousVersions"] = 0.0
        df["MinExpertiseInVersion"] = 0
        df["MinExpertiseInPreviousVersions"] = 0
        df["AvgTimeBetweenCommitsInVersion"] = None
        df["AvgTimeBetweenCommitsInPreviousVersions"] = None

        for index, row in df.iterrows():
            target_file_name = row["FileName"]
            try:
                print(f"\n\nVersion Metrics for {target_file_name} in version <= {df_version}")
                metrics = collect_metrics(hive_repo, version_commits, target_file_name.strip(), df_version)
                display_metrics(metrics)

                df.loc[index, "CommitsAffectingFileInCurrentVersion"] = metrics[df_version]["num_commits_in_version"]
                df.loc[index, "CommitsFixingBugInFileInCurrentVersion"] = metrics[df_version]["num_bug_fix_commits"]
                df.loc[index, "CommitsAffectingFileInPreviousVersions"] = metrics[df_version]["num_commits_in_previous_versions"]
                df.loc[index, "NumDevelopersInVersion"] = metrics[df_version]["num_developers_in_version"]
                df.loc[index, "NumDevelopersInPreviousVersions"] = metrics[df_version]["num_developers_in_previous_versions"]
                df.loc[index, "AvgExpertiseInVersion"] = metrics[df_version]["avg_expertise_in_version"]
                df.loc[index, "AvgExpertiseInPreviousVersions"] = metrics[df_version]["avg_expertise_in_previous_versions"]
                df.loc[index, "MinExpertiseInVersion"] = metrics[df_version]["min_expertise_in_version"]
                df.loc[index, "MinExpertiseInPreviousVersions"] = metrics[df_version]["min_expertise_in_previous_versions"]
                df.loc[index, "AvgTimeBetweenCommitsInVersion"] = metrics[df_version]["avg_time_between_commits_in_version"]
                df.loc[index, "AvgTimeBetweenCommitsInPreviousVersions"] = metrics[df_version]["avg_time_between_commits_in_previous_versions"]
            except Exception as e:
                print(f"Error processing {target_file_name}: {e}")

        df.to_csv(file, index=False)
        print(f"=== Updated file saved as {file} ===\n")
    print("\n\n\nCommit version processing successful\n\n\n")

