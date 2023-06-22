import os
import random

"""
Load random sample (N=5000)
"""

path = 'C:/Users/rojot/OneDrive/Escritorio/bd2/repo/lab10/lfw'  # replace

faces = []

for subdir, dirs, files in os.walk(path):
    for file in files:
        face_path = os.path.join(subdir, file)
        faces.append(face_path)

faces_to_compare = random.sample(faces, 5000)

print(faces_to_compare)

"""
Calculate embeddings within samples
"""
