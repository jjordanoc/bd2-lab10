from random import randint
from timeit import default_timer as timer
from typing import Tuple, List

import rtree
from rtree import index


def euclidiana(q, doc):
    dist = 0
    for i in range(len(q)):
        dist += abs(q[i] - doc[i]) ** 2
    return dist ** (1 / 2)


def query_rtree(pts: List[Tuple[int, int]], idx: rtree.Index, q: Tuple[int, int]) -> float:
    s = 0
    for k in [3, 6, 9]:
        start = timer()
        lres = list(pts[i] for i in idx.nearest(coordinates=q, num_results=k))
        end = timer()
        s += float(end - start)
        print(f"Los {k} vecinos mas cercanos usando rtree de {q} son: {lres}")
    return s / 1000


def query_linear(pts: List[Tuple[int, int]], q: Tuple[int, int]) -> float:
    s = 0
    k = 8
    dists = []
    start = timer()
    for c in pts:
        dist = euclidiana(q, c)
        dists.append([round(dist, 3), c])
    dists = sorted(dists, key=lambda x: x[0], reverse=False)
    dists = [pt for dist, pt in dists[:k]]
    end = timer()
    s += float(end - start)
    print(f"Los {k} vecinos mas cercanos usando lineal de {q} son: {dists}")

    return s / 1000


def analyze_performance(n: int):
    p = index.Property()
    p.dimension = 2  # D
    p.buffering_capacity = 10  # M
    idx = index.Index(properties=p)

    pts = [(randint(1, n), randint(1, n)) for i in range(n)]

    # insertar puntos
    for i in range(n):
        idx.insert(i, pts[i])

    # hacer consulta
    q = (1, 2)
    res_rtree = query_rtree(pts, idx, q)
    res_linear = query_linear(pts, q)
    print(f"Tiempo RTree para N={n}: {res_rtree}")
    print(f"Tiempo linear scan para N={n}: {res_linear}")


def main():
    for N in [pow(10, i) for i in range(2, 8)]:
        analyze_performance(N)


main()
