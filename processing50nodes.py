# Chemin du fichier source
fichier_source = "data/tsp50_concorde.txt"

# Chemin du nouveau fichier
nouveau_fichier = "data/nouveau_fichier.txt"

# Nombre de lignes à extraire
nombre_lignes_a_extraire = 10000

# Ouvrir le fichier source en mode lecture
with open(fichier_source, 'r') as f_source:
    # Lire toutes les lignes du fichier source
    lignes = f_source.readlines()

# Extraire les premières 10000 lignes
lignes_a_extraire = lignes[:nombre_lignes_a_extraire]

# Ouvrir le nouveau fichier en mode écriture
with open(nouveau_fichier, 'w') as f_nouveau:
    # Écrire les lignes extraites dans le nouveau fichier
    f_nouveau.writelines(lignes_a_extraire)

print(f"{nombre_lignes_a_extraire} premières lignes ont été extraites et collées dans {nouveau_fichier}.")
