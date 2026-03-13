import os
import objaverse
# Define the directory path where .npy files are stored
directory = "../data/npys"

# Initialize an empty list to store unique identifiers (UIDs) extracted from .npy files
uids = []

# Traverse through all directories and files in the specified directory
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".npy"):
            uid = file[:-4]
            uids.append(uid)
uids = list(set(uids))
# Load 3D objects from Objaverse using the extracted UIDs
objects = objaverse.load_objects(uids=uids)
