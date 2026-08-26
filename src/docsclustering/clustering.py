"""Greedy connected-component clustering for a similarity matrix."""

from typing import List, Sequence


def clusters(names: Sequence[str], S, threshold: float) -> List[List[str]]:
    """Group names so any pair within a group has similarity >= threshold.

    Uses union-find over the thresholded similarity matrix; a document no
    other document resembles ends up in its own singleton cluster.  Cluster
    members are returned sorted alphabetically, clusters in first-seen order.
    """
    n = len(names)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return [sorted(g) for g in groups.values()]
