#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:59:57 2023

@author: barret
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# -*- coding: Latin-1 -*-
"""
Created on Wed Feb 24 13:53:55 2021

@author: paulune
"""

# Import d'autres modules, fonctions...
# ----------------------------------
import matplotlib.pyplot as plt
import numpy as np
#import csv
#    with open('capitales_etats.csv', 'r') as csvfile:
#        csvfile = csv.reader(csvfile, delimiter='\t')
#        for line in csvfile:
#            print(line)

# paramètres du problème
N = 20    # nombre de villes

# paramètres de l'algorithme de recuit simulé
T0 = 10.0
Tmin = 1e-2
tau = 1e4

# Definitions constantes et globales
# ----------------------------------


# Definitions fonctions et classes
# ----------------------------------

def dist(A,B):
#    A et B sont des points i.e. que xA(yA) est l'abcisse (l'ordonnee)
#    du pts A (idem pour B)
    dAB = np.linalg.norm(A-B)
    return dAB

def energie():
    global trajet,villes,indice_ville
    V = 0.0
#    coord = np.c_[villes[0,:],villes[1,:]]
    coord = np.c_[x[trajet],y[trajet]]
    V = np.sum(np.sqrt(np.sum((coord - np.roll(coord,-1,axis=0))**2,axis=1)))
    return V   
    
def fluctuation(i,j):
    global trajet,villes,indice_ville
    Min = min(i,j)
    Max = max(i,j)
#    villes[Min:Max] = villes[Min:Max].copy()[::-1] #inverse i.e ex: 0,1,2,3 -> 3,2,1,0    
    trajet[Min:Max] = trajet[Min:Max].copy()[::-1]
    return    

# Algortithme de Metropolis
#    metropolis(x)
#    y<-yinit ; T<-Tinit
#    faire
#        ycandidat<-y+dy
#        dE<-E(x|ycandidat)-E(x|y)
#        si exp(-dE/T)>Uniform(0,1) alors
#            y<-ycandidat
#        T<-T decroit
#    jusqu'a refroidissement
#    retourne y

def metropolis(E1,E2):
    global T
    if E1 <= E2:
        E2 = E1
    else:
        dE = E1 - E2
        U = np.random.uniform(0,1)
        if U > np.exp(-dE/T):
            fluctuation(i,j)
        else:
            E2 = E1
    return E2

#function [Ropt,jopt,Copt]=tsp_exchange_recuit(X,Y,R,I,couleur,T,L)
#  D=dist(X,Y)
#  N=size(X,"*");
#  Ropt=R
#  Copt=cout(Ropt)
#  jopt=1
#  for j=1:I
#    t1=grand(1,1,'uin',1,N)
#    t2=grand(1,1,'uin',1,N-1)
#    if t2>=t1 then t2=t2+1; end
#    R=Ropt
#    R([t1,t2])=R([t2,t1])
#    C=cout(R)
#    if (C<Copt($)) then
#      accept=%T
#    else 
#      accept=(rand()<exp(-(C-Copt)/T))
#    end
#    if accept then
#      Copt=[Copt;C];
#      jopt=[jopt;j];
#      Ropt=R;
#      draw_tsp(X,Y,jopt,Copt,Ropt,couleur)
#    end
#    T=T*L
#  end
#endfunction

def matriceDistances():
    global villes
    D = np.zeros((N,N))
    for i in range(N):
        xi = villes[0,i]
        yi = villes[1,i]
        for j in range(N):
            xj = villes[0,j]
            yj = villes[1,j]
            D[i,j] = np.sqrt((xj-xi)**2 + (yj-yi)**2)
    return D
#D=sqrt((X*ones(X')-ones(X)*X').^2+(Y*ones(Y')-ones(Y)*Y').^2);


def coutAMinimiser(D):
    L = [] #liste des distances entre chaque ville 
    C =np.zeros(N)      
    for i in range(N):
        C[i] = 0.0
        for j in range(N):
            C[i] += D[i,j]
        L.append(C[i])
#    C += C+D
    return L










# Auto-test du module
# ----------------------------------
if __name__ == '__main__':
    
    print('\ttest voyageur de commerce par recuit simulé\n')
    
    # initialisation des listes d'historique
    Henergie = []     # énergie
    Htemps = []       # temps
    HT = []           # température

    x = np.random.uniform(size=N) # ne pas oublier 
    y = np.random.uniform(size=N)
    villes = np.array([x,y])
   
# Definition du trajet initial: ordre croissant des villes
#    indice_ville = np.arange(N)
#    indice_ville_init = indice_ville.copy()  
    trajet = np.arange(N)
    trajet_init = trajet.copy()


# boucle principale de l'algorithme de recuit simulé
    Ec = energie()
#    Ec = energieTotale()
    t = 0
    T = T0
    
    while T > Tmin:
        # choix de deux villes différentes au hasard
        i = np.random.randint(0,N-1)
        j = np.random.randint(0,N-1)
        if i == j: 
            continue    
        # création de la fluctuation et mesure de l'énergie
        fluctuation(i,j) 
        Ef = energie() 
        Ec = metropolis(Ef,Ec)   
    #    # application de la loi de refroidissement    
        t += 1
        T = T0*np.exp(-t/tau)
        # historisation des données
        if t % 10 == 0:
            Henergie.append(Ec)
            Htemps.append(t)
            HT.append(T)

    
    
    
    
#    **** AFFICHAGES ****  
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(x[trajet_init],
             y[trajet_init],
             'k')
    plt.plot([x[trajet_init[-1]],x[trajet_init[0]]],
         [y[trajet_init[-1]],y[trajet_init[0]]],
         'k')
    plt.plot(x,y,'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('trajet initial')
    plt.show()
    
    plt.figure(2)
    plt.subplot(1,2,2)
    plt.plot(x[trajet],
             y[trajet],
             'k')
    plt.plot([x[trajet[-1]],x[trajet[0]]],
         [y[trajet[-1]],y[trajet[0]]],
         'k')
    plt.plot(x,y,'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('trajet optimal')
    plt.show()
  
    # affichage des courbes d'évolution
    fig3 = plt.figure(3)
    plt.subplot(1,2,1)
    plt.semilogy(Htemps, Henergie)
    plt.title("Evolution de l'energie totale du systeme")
    plt.xlabel('Temps')
    plt.ylabel('Energie')
    plt.subplot(1,2,2)
    plt.semilogy(Htemps, HT)
    plt.title('Evolution de la temperature du systeme')
    plt.xlabel('Temps')
    plt.ylabel('Temperature')
    
    
    #    Affichage de l'energie en fonction du nombre de villes
    plt.figure(4)
    plt.figure(figsize=(5,3))
    D = matriceDistances()
    L = coutAMinimiser(D)
    print("{}".format(np.min(L)))
    plt.plot(L)
    plt.xlabel('N')
    plt.ylabel('L(N)')
    plt.grid()
    plt.title('L = f(N)')
    plt.show()
    
    
    
    
  
    

    
    
##    GENERALISATION
#    #    Repartition aleatoire des N villes(x,y) sur le domaine [0..1]x[0..1]
#    N = 10
#    x = np.random.uniform(size=N)
#    y = np.random.uniform(size=N)
#    ville = np.array([x,y])
#
#    
##    Definition du trajet initial: ordre croissant des villes
#    indice_ville = np.arange(N)
#    indice_ville_init = indice_ville.copy()
#    
##    Affichage du reseau
#    plt.figure(figsize=(8,5))
#    plt.plot(x[indice_ville_init],y[indice_ville_init],color='r')
#    plt.plot(x[indice_ville],y[indice_ville],color='g')
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.grid()
#    plt.title('trajet initial')
#    plt.show()

    