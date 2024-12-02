#!/usr/bin/env python
# coding: utf-8

# # UND Data Collection
# We can collect UND data. We'll use the below `run_command` fonction to run Understand via the CLI to speed up the process

# In[1]:


import pandas as pd
import sys
import os
import re
from pathlib import Path
import os
import csv
import subprocess


# In[ ]:


project_repo = "/home/nicolas-richard/Desktop/.Apache_Hive_Bug_Prediction_ML_Model/"
hive_repo = "/home/nicolas-richard/Desktop/.Apache_Hive/"
version_file = os.path.join(project_repo, "Hive_Last_Commits.csv")
und_base = os.path.join(project_repo, "UND_projects")
settings_file_path = os.path.join(project_repo, "settings.xml")
hive_data = os.path.join(project_repo, 'UND_hive_data')

os.chdir(project_repo)

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        return True  
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.cmd}")
        return False 

run_command(f'mkdir -p -m 777 {und_base}')
run_command(f' mkdir -p -m 777 {hive_data}')

os.chdir(hive_repo)

def process_versions():

    os.chdir(hive_repo)

    with open(version_file, "r") as file:
        next(file)
        for line in file:
            if line.strip():
                version, commit_id = line.split(",")[0].strip(), line.split(",")[1].strip()              

                run_command(f"cd {hive_repo}")
                run_command("git reset --hard")
                run_command("git clean -fdx")

                if not run_command(f"git checkout -f {commit_id}"):
                    sys.exit(1)
                print(f"Successfully checked out {commit_id} before {version}")


                
                und_project_path = f"{und_base}/UND_{version}.und"

                run_command(f"und create -languages java C++ {und_project_path}")

                destination_settings_file = f"{und_project_path}/settings.xml"

                if os.path.exists(destination_settings_file):
                    print(f"Removing existing settings.xml at {destination_settings_file}")
                    os.remove(destination_settings_file)

                run_command(f"cp {settings_file_path} {und_project_path}")
                
                # Redudancy here is to override SciTools Understand's automatic generation of files
                run_command(f"cp {settings_file_path} {destination_settings_file}")
                run_command(f"und settings -metricsOutputFile {os.path.join(hive_data, f'UND_{version}.csv')} {und_project_path}")

                run_command(f"und add {hive_repo} {und_project_path}")
                run_command(f"cp {settings_file_path} {destination_settings_file}")
                run_command(f"und settings -metricsOutputFile {os.path.join(hive_data, f'UND_{version}.csv')} {und_project_path}")

                run_command(f"und analyze --threads {und_project_path}")
                run_command(f"cp {settings_file_path} {destination_settings_file}")
                run_command(f"und settings -metricsOutputFile {os.path.join(hive_data, f'UND_{version}.csv')} {und_project_path}")

                run_command(f"und metrics {und_project_path}")
               
if __name__ == "__main__":
    process_versions()

