import time
from math import radians
import random
from PIL import Image, ImageDraw
from math import pi
import numpy as np
import argparse
import sys


class Labyrinthe:
    def __init__(self, nb_layer, nb_section):
        self.nb_layer = nb_layer
        self.nb_section = nb_section
        # angle d'une case du premier cercle concentrique
        self.delta = (360/self.nb_section)

        # initialisation des cercles concentrique
        self.layers = []
        self.nb_case = self.init_layer()

        # génération du graphe de proximité
        self.graph = []
        self.init_graph()
        print(self.graph)

        # création des chemins
        self.walls = []

    def init_layer(self):
        """
        Initialise les cercles concentriques avec les listes des numéros des noeuds

        :return: le nombre total de noeud(cases) dans le labyrinthe
        :rtype: int
        """
        self.nb_layer = self.nb_layer

        # nombre de section pour un cercle concentrique n
        nb_section_tmp = self.nb_section
        # nombre total de case actuel
        nb_case = 0

        # pour chaque cercle concentrique
        for n in range(1, self.nb_layer):
            # génération de la ligne n
            tetha = (180*np.arccos(((2*n**2)-1)/(2*n**2)))/pi

            if tetha <= self.delta/2:
                # on double le nombre de cases quand la taille de la case devient trop grande
                nb_section_tmp = nb_section_tmp * 2
                self.delta = tetha

            self.layers.append([nb_case+i for i in range(nb_section_tmp)])
            nb_case += nb_section_tmp
        return nb_case

    def init_graph(self, case_concentrique=5, case_non_concentrique=20):
        """
        initialise le graph d'adjacence avec des poids aléatoire pour par la suite créer un chemin aléatoire

        :param case_concentrique: paramêtre pour le random des poids des cases d'un même cercle concentrique
        :type case_concentrique: int
        :param case_non_concentrique: paramêtre pour le random des poids des cases de cercle concentrique différent
        :type case_non_concentrique: int
        """

        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                # toutes les couches sauf la dernière

                for j in range(len(self.layers[i])):
                    if len(self.layers[i+1]) > len(self.layers[i]):
                        # quand le cercle concentrique superieur est plus grand que le précédent
                        self.graph.append([
                            self.layers[i][j],
                            self.layers[i + 1][j * 2],
                            random.randint(0, case_non_concentrique)
                        ])
                        """
                        # noeud droite avec noeud milieu supérieur
                        self.graph.append([
                            self.layers[i][j],
                            self.layers[i + 1][(j * 2) + 1],
                            random.randint(0, case_non_concentrique)
                        ])
                        # noeud gauche avec noeud milieu supérieur
                        self.graph.append([
                            self.layers[i][j],
                            self.layers[i + 1][(j * 2) - 1],
                            random.randint(0, case_non_concentrique)
                        ])
                        """
                    else:
                        # quand le cercle concentrique superieur est égal au précédent
                        self.graph.append([
                            self.layers[i][j],
                            self.layers[i + 1][j],
                            random.randint(0, case_non_concentrique)
                        ])

            # les cases les unes à coté des autre
            for j in range(len(self.layers[i])):
                self.graph.append([
                    self.layers[i][j],
                    self.layers[i][(j+1) % len(self.layers[i])],# modulo comme c'est un cercle pour que ça fasse le tour
                    random.randint(0, case_concentrique)
                ])

    def make_walls(self, gen=None):
        """
        Génère les murs du labyrinthe en fonction de la class gen passé en paramètre
        Par défaut la classe pour générer les murs est Kruskal

        :param gen: classe possédent une fonction make_walls()
        :type gen: class
        """
        if not gen:
            gen = Kruskal(self.graph, self.nb_case)
            print("kruskal")

        self.walls = gen.make_walls()

    def cartesien(self, node, image_size, size=50):
        """
        calcule les coordonnées cartesiennes d'un noeud

        :param node: le noeud dont on veut la coordonnée cartésienne
        :param image_size: taille de l'image, pour calculer le la position par rapport au centre
        :return: coordonnées x et y
        :rtype: (int, int)
        """

        # récuperation du numéro de l'anneau concentrique du noeud node
        ring_number, ring_size = 0, 1
        ring_sizes = [len(layer) for layer in self.layers]

        node_tmp = node
        for i in range(len(ring_sizes)):
            if node_tmp >= ring_sizes[i]:
                ring_number += 1

                node_tmp = node_tmp - ring_sizes[i]
            else:
                break

        ring_size = len(self.layers[ring_number])
        node = node - self.layers[ring_number][0]

        # coordonnées polaire
        tetha = (360/ring_size) * node

        rayon = size + ring_number * size

        # coordonnées cartésienne
        x = rayon * np.cos(radians(tetha))
        y = rayon * np.sin(radians(tetha))

        return x + image_size//2, y + image_size//2

    def draw(self, image_name, image_size=400, lab_size=50):
        """
        Dessine, génère l'image du labyrinthe circulaire

        :todo: passer la sortie en svg

        :param image_name: nom de l'image de sortie
        :type image_name: str
        :param image_size: taille de l'image voulu
        :type image_size: int
        :return: image au format Pillow
        """

        # x = y = image_size*self.nb_layer
        image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for edge in self.walls:
            (n1, n2) = edge
            if n2 < n1:
                n2, n1 = n1, n2

            c_n1 = self.get_layer(n1)
            c_n2 = self.get_layer(n2)

            # calcul de n1 dans les cas spéciaux pour le placement d'un trait
            # avec comme coordonné le milieu en n1 + ou - 1 modulo nombre de section et n1
            if len(self.layers[c_n1]) < len(self.layers[c_n2]):
                if self.nb_section % 2:
                    if not n2 % 2:
                        o_n1 = self.find_other_graph(n1, n2)[0]
                        # milieu o_n1 n1
                        if o_n1 == (n1-1) % len(self.layers[c_n1]):
                            n1 = (n1 - 0.5) % len(self.layers[c_n1])
                        else:
                            n1 = (n1 + 0.5) % len(self.layers[c_n1])
                else:
                    if n2 % 2:
                        o_n1 = self.find_other_graph(n1, n2)[0]
                        # milieu o_n1 n1
                        if o_n1 == (n1 - 1) % len(self.layers[c_n1]):
                            n1 = (n1 - 0.5) % len(self.layers[c_n1])
                        else:
                            n1 = (n1 + 0.5) % len(self.layers[c_n1])

            x1, y1 = self.cartesien(n1, image_size, lab_size)
            x2, y2 = self.cartesien(n2, image_size, lab_size)

            draw.line((x1, y1, x2, y2),
                      fill=(0, 0, 0), width=1)

        del draw
        image.save('{}.png'.format(image_name))

    def find_other_graph(self, n1, n2):
        """
        Retourne le couple u1,u2 voisin de n1,n2 avec u2==n2

        :param n1: noeud du graph
        :type n1: int
        :param n2: noeud du graph
        :type n2: int
        :return: couple de noeud
        :rtype: (int, int)
        """
        c_n1 = self.get_layer(n1)
        lc_n1 = len(self.layers[c_n1])
        print("n1, n2 : ", (n1, n2))
        for edge in self.graph:
            u1, u2, w = edge
            if u1 == n2:
                if u2 == ((n1-self.layers[c_n1][0] - 1) % lc_n1)+self.layers[c_n1][0] or u2 == ((n1-self.layers[c_n1][0] + 1) % lc_n1)+self.layers[c_n1][0]:
                    return (u2, u1)
            elif u2 == n2:
                print("bonjour")
                print(self.layers[c_n1][0])
                print("(n1 - 1)", (n1-self.layers[c_n1][0] - 1) % lc_n1)
                print("(n1 + 1)", (n1-self.layers[c_n1][0] + 1) % lc_n1)
                if u1 == ((n1-self.layers[c_n1][0] - 1) % lc_n1)+self.layers[c_n1][0] or u1 == ((n1-self.layers[c_n1][0] + 1) % lc_n1)+self.layers[c_n1][0]:
                    print("aurevoir")
                    return (u1, u2)
        return ()

    def get_layer(self, node):
        """
        Retourne le numéro du cercle concentrique de noeud

        :param node: un numéro de noeud du graph
        :type node: int
        :return: le numéro de la couche
        :rtype: int
        """
        for i_layer, layer in enumerate(self.layers):
            if node in layer:
                return i_layer
        return 0


class Kruskal:
    """
        https://fr.wikipedia.org/wiki/Algorithme_de_Kruskal#Pseudo-code
        https://fr.wikipedia.org/wiki/Union-find#Principe

    """
    def __init__(self, graph, nb_noeud):
        self.graph = graph
        self.nb_noeud = nb_noeud
        self.parent = {}
        self.rank = {}
        for i in range(self.nb_noeud):
            self.parent[i] = i
            self.rank[i] = 0

    def make_set(self, node):
        self.parent[node] = None

    def find(self, noeud):
        if self.parent[noeud] != noeud:
            self.parent[noeud] = self.find(self.parent[noeud])
        return self.parent[noeud]

    def union(self, node1, node2):
        xroot = self.find(node1)
        yroot = self.find(node2)

        if xroot != yroot:
            self.parent[xroot] = yroot

    def make_walls(self):
        walls = []

        # trier la liste des arrètes en fonction des poids
        self.graph = sorted(self.graph, key=lambda item: item[2])

        for edge in self.graph:
            node1, node2, weigth = edge
            if self.find(node1) != self.find(node2):
                walls.append((node1, node2))
                self.union(node1, node2)

        return walls


if __name__ == "__main__":

    path = "lab2"

    #créer un parser en ligne de commande 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--o', help='Name of the image', type=str, default=None)
    parser.add_argument('-nbc', '--nbc', help='Number of circles', nargs='?', const=1, type=int, required=True)
    parser.add_argument('-nbs', '--nbs', help='Number of section of the center circle', nargs='?', const=1, type=int, required=True)
    parser.add_argument('-nbi', '--nbi', help='Number of image when want to generate', nargs='?', const=1, type=int, default=1)
    args = parser.parse_args()

    for i in range(args.nbi):
        if not args.o:
            timestamp = str(time.strftime("%Y-%b-%d_%H-%M-%S", time.gmtime(time.time())))
            path = timestamp + "-" + str(i)
        else:
            path = args.o

        seed = str(random.randrange(sys.maxsize))
        random.seed(seed)

        l = Labyrinthe(args.nbc, args.nbs)

        l.make_walls(Kruskal(l.graph, l.nb_case))
        l.draw(path, 500, 50)

        with open(path, "w") as seed_file:
            seed_file.write(seed)



