import numpy as np

def beam_search(probs, beam_width=5):
    batch_size, n_nodes, _ = probs.shape
    
    # Initialisation des états
    states = np.zeros((batch_size, beam_width, 1), dtype=int)
    scores = np.zeros((batch_size, beam_width))
    
    for i in range(n_nodes):
        # Étendre les états actuels
        new_states = np.repeat(states, n_nodes, axis=1)
        
        # Sélectionner les probabilités associées aux transitions possibles
        transition_probs = np.choose(new_states[:, :, -1], probs.transpose(0, 2, 1))
        
        # Calculer les nouveaux scores
        new_scores = scores + np.log(transition_probs)
        
        # Mettre à jour les indices des états
        indices = np.argsort(new_scores, axis=1)[:, -beam_width:]
        indices_col = indices % n_nodes
        indices_row = np.arange(batch_size).reshape(-1, 1)
        states = np.concatenate([new_states[indices_row, indices_col, :], indices.reshape(batch_size, beam_width, 1)], axis=2)
        
        # Mettre à jour les scores
        scores = np.take_along_axis(new_scores, indices, axis=1)
    
    # Sélectionner le chemin avec le score le plus élevé
    best_path_index = np.argmax(scores[:, -1])
    best_path = states[best_path_index]
    
    return best_path[0, 1:]  # Retourner le chemin sans le premier élément (qui est généralement 0)

# Exemple d'utilisation
batch_size = 2
n_nodes = 3
probs = np.random.rand(batch_size, n_nodes, n_nodes)
best_paths = beam_search(probs, beam_width=3)
print("Meilleurs chemins:", best_paths)
