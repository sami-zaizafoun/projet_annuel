Avant d'exécuter les programmes, veuillez lancer la commande suivante afin d’être sûr d’avoir tous les librairies utilisées: 
	
------------------------------
pip install -r requirement.txt 
------------------------------

Fisheye:
	Pour lancer le programme fisheye.py, il faut lancer la commande suivante:

		fisheye.py -iFile ...  -oFile ... -r ... -c ... -d ...

			-iFile ou --iFile : Le nom du fichier d'entrée -> obligatoire
			-oFile ou --oFile : Le nom du fichier de sortie -> obligatoire
    			-r ou --r : Rayon de l'image -> valeur par défaut défini
    			-c ou --c : Centre de l'image -> valeur par défaut défini
    			-d ou --d : démagnification de l'image -> valeur par défaut défini

Labyrinthe:
	Pour lancer le programme labyrinthe.py, il faut lancer la commande suivante:

		labyrinthe.py -o ... -nbc ...  -nbs ... -nbi ... 
			
			-o ou --o : Le nom de l'image -> valeur par défaut défini
			-nbc ou --nbc : Le nombre des cercles -> obligatoire
			-nbs ou --nbs : Le nombre des sections du cercle concentrique -> obligatoire
			-nbi ou --nbi : Le nombre d'image(s) à générer -> valeur par défaut défini

  
Exemple utilisation (windows):
	
	cd src
	python labyrinthe.py -o ..\ressource\labtest -nbc 8 -nbs 6
	python fisheye.py -iFile ..\ressource\labtest.png -oFile ..\ressource\labtestfisheye.png
	
	
