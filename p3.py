from timeit import default_timer as timer
from typing import Tuple, List
import rtree
from rtree import index
import os
import random
import face_recognition
import numpy as np

from Heap import MaxHeap

path = 'C:/Users/rojot/OneDrive/Escritorio/bd2/repo/lab10/lfw'  # replace
k = 8  # always find 8 nearest


def query_rtree(embeds: List[np.ndarray], idx: rtree.Index, q: np.ndarray) -> float:
    s = 0
    start = timer()
    lres = list(embeds[i] for i in idx.nearest(coordinates=q, num_results=k))
    end = timer()
    s += float(end - start)
    return s / 1000


def query_linear(pts: List[np.ndarray], q: np.ndarray) -> float:
    class DistWrapper:
        def __init__(self, d: float, embed: np.ndarray):
            self.dist = d
            self.embed = embed

        def __lt__(self, other):
            return self.dist < other.dist

        def __gt__(self, other):
            return self.dist > other.dist

        def __eq__(self, other):
            return self.dist == other.dist

    s = 0
    dists = MaxHeap[DistWrapper]()
    start = timer()
    for c in pts:
        dist = np.linalg.norm(c - q)
        if dists.size() < k:
            dists.push(DistWrapper(d=dist, embed=c))
        elif dists.top().dist > dist:
            dists.pop()
            dists.push(DistWrapper(d=dist, embed=c))
    end = timer()
    s += float(end - start)

    return s / 1000


def analyze_performance(n: int):
    p = index.Property()
    p.dimension = 128  # D
    p.buffering_capacity = 10  # M
    idx = index.Index(properties=p)

    faces = []

    for subdir, dirs, files in os.walk(path):
        for file in files:
            face_path = os.path.join(subdir, file)
            faces.append(face_path)

    faces_to_compare = random.sample(faces, n)

    face_embeddings = []

    for face_file in faces_to_compare:
        image = face_recognition.load_image_file(face_file)
        embedding_list = face_recognition.face_encodings(image)
        if len(embedding_list) > 0:
            face_embeddings.append(embedding_list[0])

    # insertar puntos
    for i in range(len(face_embeddings)):
        idx.insert(i, face_embeddings[i])

    # hacer consulta
    q = face_embeddings[5]
    res_rtree = query_rtree(face_embeddings, idx, q)
    res_linear = query_linear(face_embeddings, q)
    print(f"Tiempo RTree para N={n}: {res_rtree}")
    print(f"Tiempo linear scan para N={n}: {res_linear}")


def main():
    for N in [pow(10, i) for i in range(1, 4)]:
        analyze_performance(N)


main()
