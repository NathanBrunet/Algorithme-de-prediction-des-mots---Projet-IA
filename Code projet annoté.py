txt_brut = ['texte initial d_entrée']

mots = []
for text in txt_brut: #On met tous les mots dans une liste
    for mot in text.split(' '):
        mots.append(mot)

mots = set(mots) #On supprime les doublons

word2int = {}

for i,mot in enumerate(mots):
    word2int[mot] = i

phrases = []
for phrase in txt_brut:
    phrases.append(phrase.split()) #On sépare les phrases pour qu'un mot d'une même phrase ne soit pas pris comme voisin d'un mot d'une autre phrase

taille_fenetre = 2

data = []
for phrase in phrases:
    for idx, mot in enumerate(phrase):
        for vois in phrase[max(idx - taille_fenetre, 0) : min(idx + taille_fenetre, len(phrase)) + 1] : #Afin de ne pas être en indice négatif ou supérieur à la longueur de la phrase
            if vois != mot:
                data.append([mot, vois]) #" data " va contenir des listes de 2 mots avec toutes les combinaisons de 2 mots dans une fenêtre de 2 autour du mot
               

import pandas as pd

df = pd.DataFrame(data, columns = ['input', 'label']) #Mettre sous forme de tableau 

import tensorflow as tf
import numpy as np

ONE_HOT_DIM = len(mots) #La dimension des vecteurs mots sera du nombre de mots que contient le dictionnaire

# Function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding  # Fonction servant à mettre en vecteurs les mots qui ont été indexés lignes 22-23 word2int:mot --> entier

X = [] # Input word
Y = [] # Target word

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ])) #Ici on met sous forme de vecteur avec un seul 1 et que des zéros les mots choisis
    Y.append(to_one_hot_encoding(word2int[ y ])) #Ici on met sous forme de vecteur avec un seul 1 et que des zéros leurs voisins dans une fenêtre de 2 autour

# Conversion en tableaux
X_train = np.asarray(X)
Y_train = np.asarray(Y)

# Making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM)) #Donne de l'espace pour les données
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

EMBEDDING_DIM = 2  #On met deux neurones pour pouvoir avoir 2 caractéristiques que l'on modélisera comme des coordonnées (afin de pouvoir construire un graphe)

W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM])) #Poids 1 initialisé au hasard
b1 = tf.Variable(tf.random_normal([1])) #Biais 1
hidden_layer = tf.add(tf.matmul(x,W1), b1) #On ajoute cette couche cachée


W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM])) #Poids 2 initialisé au hasard
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2)) #Prédiction faite avec la fonction d'activation softmax pour cette dernière couche 


loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1])) # Fonction log

# Training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

iteration = 20000
for i in range(iteration):
    # L'entrée est X_train : mot d'entrée dont on veut les voisins 
    # la deuxième colonne est ce que l'on veut : les voisins, et l'on va entrainer les deux listes pour avoir les bons voisins pour les entrées considérées
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train}) #On lance la session TF
    

pts = sess.run(W1 + b1) #Lancer la session TF qui va donner en sortie les liste de deux coordonnées (poids) dont on fera une probabilité grâce à un calcul de distances W1 b1 : hidden layer
print(pts)


motss=list(mots)


l=motss.index('psychology') # On prend l'indice du mot dont on veut chercher le voisin le plus probable


def probas(k): #Calcule la distance de chaque mot à un mot d'indice k dans la liste : motss puis on divise ces distances par la somme des distances afin d'obtenir une probabilité de voisinage 
    D=[]
    S=0
    for i in range (len(pts)):
        D=D+[np.sqrt((pts[i][1]-pts[k][1])**2+(pts[i][0]-pts[k][0])**2)]
        S=S+np.sqrt((pts[i][1]-pts[k][1])**2+(pts[i][0]-pts[k][0])**2)
    for i in range (len(D)):
        D[i]=D[i]/S
        
    return D

def mini(probas):  #On calcule le minimum des probas de la fonction proba car c'est ce qui nous donnera l'élément le plus proche du mot d'entrée 
    if probas[0]!=0:
        mini=probas[0]
        indice=0
    else :
        mini=probas[1]
        indice=1
    for i in range (len(probas)) :
        if probas[i]<mini and probas[i]!=0:
            mini=probas[i]
            indice=i
    return  motss[indice] #Retourne le mot prédit

print(mini(probas(l)))