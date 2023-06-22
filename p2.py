import os
import random
import matplotlib.pyplot as plt
import face_recognition
import numpy as np

"""
Load random sample
"""

path = 'C:/Users/rojot/OneDrive/Escritorio/bd2/repo/lab10/lfw'  # replace

N = 5000

faces = []

for subdir, dirs, files in os.walk(path):
    for file in files:
        face_path = os.path.join(subdir, file)
        faces.append(face_path)

faces_to_compare = random.sample(faces, N)

"""
Calculate embeddings within samples
"""

face_embeddings = []

for face_file in faces_to_compare:
    image = face_recognition.load_image_file(face_file)
    embedding_list = face_recognition.face_encodings(image)
    print(len(embedding_list[0]))
    if len(embedding_list) > 0:
        face_embeddings.append(embedding_list[0])

"""
Calculate euclidean distance
"""

x = []

for img1_embeds in face_embeddings:
    for img2_embeds in face_embeddings:
        dist = np.linalg.norm(img1_embeds - img2_embeds)  # euclidean distance
        x.append(dist)

plt.hist(x)
plt.show()
