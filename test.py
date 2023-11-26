import numpy as np

def k_plus_proches_points(edges_dist, k):
    indices_proches = np.argsort(edges_dist, axis=1)[:, 1:k+1]
    return indices_proches

# Exemple d'utilisation
# Supposons que edges_dist soit une matrice 2D représentant les distances entre les points
# et que k soit le nombre de points les plus proches que vous souhaitez trouver.
edges_dist = np.array([[0, 1, 2, 3],
                      [1, 0, 4, 5],
                      [2, 4, 0, 6],
                      [3, 5, 6, 0]])

k = 2
indices_proches = k_plus_proches_points(edges_dist, k)

print("Indices des", k, "points les plus proches pour chaque point:")
print(indices_proches)