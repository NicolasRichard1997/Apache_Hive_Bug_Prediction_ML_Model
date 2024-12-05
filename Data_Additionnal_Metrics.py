#!/usr/bin/env python
# coding: utf-8

# # Additionnal Metrics
# As per the specification of the personnal project, we'll try and gather additionnal metrics to improve our model. Since this notebook is very compute-intensive, it was executed directly in the terminal after converting this notebook to a python file.
# ## 1. Lines added & deleted from a given version
# We'll begin by creating a dictionnary of versions and corresponding commits from the file `Hive_Last_Commits.csv`previously created

# In[19]:


import csv
import git
import glob
import logging
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Tuple
import pandas as pd


# In[20]:


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

    output_path = output_directory / f"UND_{df_version}.csv"
    df.to_csv(output_path, index=False)
    print(f"=== Updated file saved as {output_path} ===\n")


# ## 2. Commits Affecting each File, Developper Experience and Comparison with Previous Versions
# In this section, we'll define functions to generate the metrics for commits affecting the file, developpers having worked on the project and their expertise. Here are the metrics, their definition and their collection method we aim to gather for each file/version:

# #### Metrics
# 1. Number of Commits Affecting File in Version (`num_commits_in_version`)
# 
#     Definition: Total number of commits that have modified the target file in the current version.
#     Collection Method:
#         Fetch all commits that have affected the target file across all versions.
#         Map each commit to its respective version based on the commit ranges defined between version tags.
#         Count the number of commits affecting the file in the current version.
# 
# 2. Number of Bug Fix Commits in Version (`num_bug_fix_commits`)
# 
#     Definition: Number of commits in the current version that address bug fixes in the target file.
#     Collection Method:
#         From the commits affecting the file in the current version, identify commits with messages containing bug fix keywords (e.g., "fix", "bug", "issue", "HIVE-").
#         Count these bug fix commits.
# 
# 3. Number of Commits in Previous Versions (`num_commits_in_previous_versions`)
# 
#     Definition: Total number of commits that have modified the target file in all versions prior to the current version.
#     Collection Method:
#         Aggregate commits affecting the file from all previous versions.
#         Count the total number of these commits.
# 
# 4. Number of Developers in Version (`num_developers_in_version`)
# 
#     Definition: Number of unique developers who have committed changes to the target file in the current version.
#     Collection Method:
#         Extract the email addresses of authors from the commits affecting the file in the current version.
#         Count the number of unique developer emails.
# 
# 5. Number of Developers in Previous Versions (`num_developers_in_previous_versions`)
# 
#     Definition: Number of unique developers who have committed changes to the target file in all previous versions.
#     Collection Method:
#         Extract the email addresses of authors from the commits affecting the file in previous versions.
#         Count the number of unique developer emails.
# 
# 6. Average Expertise in Version (`avg_expertise_in_version`)
# 
#     Definition: The average total number of commits made by developers (across the entire repository) who have contributed to the target file in the current version.
#     Collection Method:
#         For each developer who committed to the file in the current version, retrieve their total number of commits in the entire repository (developer experience).
#         Calculate the average of these totals.
# 
# 7. Average Expertise in Previous Versions (`avg_expertise_in_previous_versions`)
# 
#     Definition: The average total number of commits made by developers who have contributed to the target file in previous versions.
#     Collection Method:
#         For each developer from previous versions, retrieve their total number of commits in the entire repository.
#         Calculate the average of these totals.
# 
# 8. Minimum Expertise in Version (`min_expertise_in_version`)
# 
#     Definition: The smallest total number of commits (in the entire repository) among developers who have contributed to the target file in the current version.
#     Collection Method:
#         Retrieve the total commits for each developer in the current version.
#         Identify the minimum total commits among them.
# 
# 9. Minimum Expertise in Previous Versions (`min_expertise_in_previous_versions`)
# 
#     Definition: The smallest total number of commits among developers who have contributed to the target file in previous versions.
#     Collection Method:
#         Retrieve the total commits for each developer in previous versions.
#         Identify the minimum total commits among them.
# 
# 10. Average Time Between Commits in Version (`avg_time_between_commits_in_version`)
# 
#     Definition: The average time (in seconds) between consecutive commits affecting the target file within the current version.
#     Collection Method:
#         Sort the commit timestamps of commits affecting the file in the current version.
#         Calculate the time differences between each pair of consecutive commits.
#         Compute the average of these time differences.
# 
# 11. Average Time Between Commits in Previous Versions (`avg_time_between_commits_in_previous_versions`)
# 
#     Definition: The average time between consecutive commits affecting the target file in all previous versions.
#     Collection Method:
#         Sort the commit timestamps of commits affecting the file in previous versions.
#         Calculate the time differences between each pair of consecutive commits.
#         Compute the average of these time differences.
# 
# 12. Number of commits to file F during version V that have changed a code comment. (`num_commits_with_comment_changes`)
# 
#     Definition: Number of commits to file F during version V that have changed a code comment.
#     Collection Method:
#         After retrieving the commits affecting the file in the current version, analyze 
#         each commit to check if it changed a code comment.
# 
# 13. Number of commits to file F during version V that have not changed a code comment. (`num_commits_without_comment_changes`)
# 
#     Definition: Number of commits to file F during version V that have not changed a code comment. 
#     Collection Method:
#         After retrieving the commits affecting the file in the current version, analyze 
#         each commit to check if it changed a code comment. The opposite metric as above.
#         It may be obtain by verifying the number of commits made in that version, and substracting the above metric

# We'll begin by defining helper functions before defining metrics collection procedure and executing it all at once in order to speed up the process and minimize read-write operations.

# In[ ]:


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


# Next, we'll have to define metrics for developper experience. to simplify this, we can define this metric as total experience for the project. Hence, we can fetch all developpers having worked on the project and assign their experience as their numbers of commits on the project. While this way of defining experience is more than imperfect, the output gives us a solid base for the trainning to come

# In[ ]:


global developer_experiences 

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
    if len(commits) >= 2:
        commit_dates = sorted([commit.committed_datetime for commit in commits])
        time_diffs = [(commit_dates[i+1] - commit_dates[i]).total_seconds() for i in range(len(commit_dates)-1)]
        avg_time_between_commits = sum(time_diffs) / len(time_diffs)
    else:
        avg_time_between_commits = None
    return avg_time_between_commits


# For the comments-related metrics, we can define the following pattern to identify whether a change to a file is a comment  

# In[ ]:


def is_comment_change(diff_text):
    """Determine if the diff includes changes to code comments."""
    # Define regex patterns for comment lines
    comment_patterns = [
        r'^\s*//',       # C++/Java style single-line comment
        r'^\s*/\*',      # Start of C-style multi-line comment
        r'^\s*\*',       # Inside C-style multi-line comment
        r'^\s*\*/',      # End of C-style multi-line comment
    ]
    pattern = re.compile('|'.join(comment_patterns))

    for line in diff_text.split('\n'):
        line = line.strip()
        if line.startswith('+') or line.startswith('-'):
            code_line = line[1:].strip()
            if pattern.match(code_line):
                return True
    return False


# As mentionned before, this task will be computationnaly very expensive. Hence, in order to minimize the quantity of git commands and read/write operations, we'll need to gather all of our metrics in batch.

# In[ ]:


def collect_metrics(hive_repo, version_commits, target_file_name, df_version):
    """Collect metrics for a target file."""
    global developer_experiences  
    repo = git.Repo(hive_repo)
    metrics = defaultdict(dict)

    relevant_versions = [vc for vc in version_commits if compare_versions(vc[0], df_version)]

    print(f"Fetching all commits affecting {target_file_name}...")
    all_commits_affecting_file = list(repo.iter_commits(paths=target_file_name.strip()))

    commit_to_version = {}
    for i, (version, commit_hash) in enumerate(relevant_versions):
        if i < len(relevant_versions) - 1:
            next_commit_hash = relevant_versions[i + 1][1]
        else:
            next_commit_hash = 'HEAD'

        commit_range = f"{commit_hash}..{next_commit_hash}"
        commits_in_range = list(repo.iter_commits(commit_range))

        for commit in commits_in_range:
            commit_to_version[commit.hexsha] = version

    version_to_commits = defaultdict(list)
    for commit in all_commits_affecting_file:
        commit_version = commit_to_version.get(commit.hexsha)
        if commit_version:
            version_to_commits[commit_version].append(commit)

    all_previous_commits = []

    for version in relevant_versions:
        version = version[0]
        commits_affecting_file = version_to_commits.get(version, [])

        # Identify bug fix commits
        bug_fix_keywords = ["fix", "bug", "issue", "HIVE-"]
        bug_fix_commits = [
            c for c in commits_affecting_file if any(keyword in c.message.lower() for keyword in bug_fix_keywords)
        ]

        commits_with_comment_changes = []
        commits_without_comment_changes = []
        for commit in commits_affecting_file:
            diffs = commit.diff(commit.parents[0] if commit.parents else None, paths=target_file_name.strip(), create_patch=True)
            comment_change = False
            for diff in diffs:
                diff_text = diff.diff.decode('utf-8', errors='ignore')
                if is_comment_change(diff_text):
                    comment_change = True
                    break
            if comment_change:
                commits_with_comment_changes.append(commit)
            else:
                commits_without_comment_changes.append(commit)

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
            "avg_time_between_commits_in_previous_versions": avg_time_between_commits_in_prev_versions,
            "num_commits_with_comment_changes": len(commits_with_comment_changes),
            "num_commits_without_comment_changes": len(commits_without_comment_changes),
        }

        all_previous_commits.extend(commits_affecting_file)

        all_previous_commits = list(set(all_previous_commits))

        if version == df_version:
            break

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
        print(f"  - Commits with comment changes in version: {data['num_commits_with_comment_changes']}")
        print(f"  - Commits without comment changes in version: {data['num_commits_without_comment_changes']}")
        print()


# Finally, we'll define an overarching loop to gather all these metrics: 

# In[ ]:


if __name__ == "__main__":
    import sys

    version_commits_file = "Hive_Last_Commits.csv"
    hive_repo = "/path/to/hive/repo"  # Update with the path to your Hive repository
    project_repo = "/path/to/project/repo"  # Update with the path to your project repository

    version_commits = load_version_commits(version_commits_file)
    files_dir = os.path.join(project_repo, "UND_hive_additional_metrics")
    files = sorted([
        os.path.join(files_dir, f) 
        for f in os.listdir(files_dir) 
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
        df["AvgTimeBetweenCommitsInVersion"] = 0
        df["AvgTimeBetweenCommitsInPreviousVersions"] = 0
        df["CommitsWithCommentChanges"] = 0
        df["CommitsWithoutCommentChanges"] = 0

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
                df.loc[index, "CommitsWithCommentChanges"] = metrics[df_version]["num_commits_with_comment_changes"]
                df.loc[index, "CommitsWithoutCommentChanges"] = metrics[df_version]["num_commits_without_comment_changes"]
            except Exception as e:
                print(f"Error processing {target_file_name}: {e}")

        df.to_csv(file, index=False)
        print(f"=== Updated file saved as {file} ===\n")
    print("\n\n\nCommit version processing successful\n\n\n")

