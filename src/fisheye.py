from __future__ import print_function

import argparse
import matplotlib.pyplot as pl
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RectBivariateSpline
import cv2


class Fisheye:

    def __init__(self, rayon, deformation, centre):
        self.centre = np.array(centre)
        self.rayon = rayon
        self.deformation = deformation

    def process_image(self, img):
        # enregistre la dimension de base de l'image
        original_shape = img.shape

        data = img.copy()
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])

        # modification des coordonnées

        # seulement les coordonnées à modifier
        cercle = self.distortion_circle()

        newcoords = self.fisheyeCoords(cercle)

        new_data = data.copy()

        for color in range(data.shape[2]):
            # a partir des nouvelles coords calculé on interpole la fonction pour calculer les blancs
            # interpolation de la fonction la taille de l'image bien sure
            interpolate_function = RectBivariateSpline(x, y, data[:, :, color], kx=1, ky=1)

            # transformation de l'image à partir de cette fonction interpolet
            transformed = interpolate_function(newcoords[:, 0].flatten(), newcoords[:, 1].flatten(), grid=False)
            new_data[:, :, color] = data[:, :, color]

            # nouvelle image où l'on ajoute le cercle de déformation
            new_data[cercle[:, 0].flatten(), cercle[:, 1].flatten(), color] = transformed

        return new_data.reshape(original_shape)

    def fisheyeCoords(self, pos, inverse=False):
        data = pos.copy()

        # tetha coordonnées polaire
        theta = self.calcul_theta(pos)

        # ratio de distortion par rapport au centre du cercle
        ratios = self.ratio_distortion(data)

        # taux de distortion du fisheye
        taux = ratios.copy()
        if not inverse:
            taux[ratios < 1] = self.taux_distortion(ratios)
        else:
            taux[ratios < 1] = self.taux_distortion_inverse(ratios)

        # tableau contenant les nouvelle coordonée
        newcoords = np.empty_like(pos)

        # nouveau x
        newcoords[:, 0] = self.centre[0] + np.cos(theta) * self.rayon * taux
        # nouveau y
        newcoords[:, 1] = self.centre[0] + np.sin(theta) * self.rayon * taux

        return newcoords

    def distortion_circle(self):
        # ensemble des coordonnées du cercle de distortion
        coords = []

        # pour i allant du debut(centre x - rayon) du cercle de distortion jusqu'à la fin(centre x + rayon)
        for i in range(self.centre[0] - self.rayon, self.centre[0] + self.rayon):
            # pour j allant du debut(centre y - rayon) du cercle de distortion jusqu'à la fin(centre y + rayon)
            for j in range(self.centre[1] - self.rayon, self.centre[1] + self.rayon):

                # si le point est dans le rayon du cercle
                if cdist(np.array([i, j]).reshape(1, 2), self.centre.reshape(1, 2)) < self.rayon:
                    coords.append((i, j))
        return np.array(coords)

    def ratio_distortion(self, coords):
        return cdist(coords, self.centre.reshape(1, 2)).flatten() / self.rayon

    def taux_distortion(self, ratios):
        return (ratios[ratios < 1])/(self.deformation*(1 - ratios[ratios < 1]) + 1)

    def taux_distortion_inverse(self, ratios):
        return (ratios[ratios < 1] + self.deformation * ratios[ratios < 1]) / ((self.deformation * ratios[ratios < 1])+1)

    def calcul_theta(self, coords):
        theta = []
        for (y, x) in coords:
            theta.append(np.arctan2(y - self.centre[1], x - self.centre[0]))
        return theta

    def set_centre(self, x, y):
        self.centre[0], self.centre[1] = x, y


def grid_plot():
    deformation = [1.5, 3, 4.5, 6]

    nb_line = 15
    rayon = 0.4

    fig, ax = pl.subplots(len(deformation), 1, figsize=(2, 5))

    dir1 = np.linspace(0, 1, 500)

    for index_d, d in enumerate(deformation):
        f = Fisheye(rayon, d, np.array([0.5, 0.5]))

        for line in range(1, nb_line):
            dir2 = np.ones_like(dir1) * (1.0/nb_line) * line

            pos_1 = np.empty((len(dir1), 2))
            pos_2 = np.empty((len(dir1), 2))
            pos_1[:, 0] = dir1
            pos_1[:, 1] = dir2
            pos_2[:, 0] = dir2
            pos_2[:, 1] = dir1

            pos_1 = f.fisheyeCoords(pos_1, inverse=True)
            pos_2 = f.fisheyeCoords(pos_2, inverse=True)

            ax[index_d].plot(pos_1[:, 0], pos_1[:, 1], '-k', lw=0.5)
            ax[index_d].plot(pos_2[:, 0], pos_2[:, 1], '-k', lw=0.5)

    fig.tight_layout()
    pl.subplots_adjust(wspace=0.01, hspace=0.01)
    pl.show()


def test_image(filename):
    img = cv2.imread(filename)
    centre = [img.shape[0] // 2, (img.shape[1] // 2)]
    rayon = centre[0]#//4

    f = Fisheye(rayon, 4, centre)

    new_img = f.process_image(img)

    cv2.imshow("test", new_img)
    cv2.waitKey()


def make_bigger_image(image, output_width):
    new_image = np.ones(output_width)*255

    sha = image.shape
    height = sha[0]//2
    width = sha[1]//2
    x_range = np.arange(start=(output_width[0]//2) - height, stop=(output_width[0]//2) + height)
    y_range = np.arange(start=(output_width[1]//2) - width, stop=(output_width[1]//2) + width)

    for x_d, x in enumerate(x_range):
        new_image[x, y_range] = image[x_d]

    cv2.imshow("test", new_image)
    cv2.waitKey()

    rayon = np.sqrt((sha[0]**2)+(sha[1]**2))
    rayon = rayon.astype(int)
    centre = (new_image.shape[0]//2, new_image.shape[1]//2)
    f = Fisheye(rayon, 4, centre)

    new_img = f.process_image(new_image)

    cv2.imshow("test", new_img)
    cv2.waitKey()


if __name__ == "__main__":    

    #créer un parser en ligne de commande 
    parser = argparse.ArgumentParser()    
    parser.add_argument('-iFile', '--iFile', help='Input file name', type=str, required=True)    
    parser.add_argument('-r', '--r', help='Image radius',nargs='?', const=1, type=int)
    parser.add_argument('-c', '--c', help='Image center',nargs='+', type=int)
    parser.add_argument('-d', '--d', help='Image deformation rate',nargs='?', const=1, type=float)    
    parser.add_argument('-oFile', '--oFile', help='Output file name', type=str, required=True)
    args = parser.parse_args()

    img = cv2.imread(args.iFile)

    #Valeurs par défaut
    if not args.r:
        args.r = min(img.shape[0],img.shape[1])//2    
    if not args.d:
        args.d = 4
    if not args.c:
        args.c = [img.shape[0]//2,(img.shape[1]//2)]

    print("Input image:", args.iFile, "\nOutput image:", args.oFile,"\nRayon=", args.r,"\nDeformation=", args.d,"\nCentre=", args.c)    
    f = Fisheye(args.r, args.d, args.c)    

    new_img = f.process_image(img)
    cv2.imwrite(args.oFile,new_img)

    # grid_plot()
