# -*- coding:utf-8 -*-

import itertools
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T
from theano import function

import matplotlib.pyplot as plt

import cPickle
import gzip


def cast_data(data):
    """
    Cast les données en floatX et int32
    X : floatX
    y : int32

    Returns
    -------
    liste de tuples representant (x, y)
    """
    return [(x.astype(theano.config.floatX), y.astype('int32')) for x, y in data]


def load_2moons():
    """
    Charge les données de l'ensemble 2moons

    Returns
    -------
    list de tuples representant (x, y) pour l'ensemble d'entraînement 
    et l'ensemble de test
    """
    data = np.loadtxt("2moons.txt")

    train_x = data[:800,:-1]
    train_y = data[:800,-1]
    test_x = data[:800,:-1]
    test_y = data[:800,-1]

    return cast_data([(train_x, train_y), (test_x, test_y)])


def load_mini_mnist():
    """
    Charge la version mini de l'ensemble MNIST 
    (1000 examples d'entraînement)

    Returns
    -------
    list de tuples representant (x, y) pour l'ensemble d'entraînement,
    de validation et l'ensemble de test
    """
    train_x = np.loadtxt("train_images.txt", delimiter=",")
    train_y = np.loadtxt("train_labels.txt", delimiter=",").argmax(1)
    x = np.loadtxt("test_images.txt", delimiter=",")
    y = np.loadtxt("test_labels.txt", delimiter=",")
    valid_x = x[:500]
    valid_y = y[:500].argmax(1)
    test_x = x[500:]
    test_y = y[500:].argmax(1)

    data = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

    return cast_data(data)


def load_medium_mnist():
    """
    Charge la version moyenne de l'ensemble MNIST
    (5000 examples d'entraînement)

    Returns
    -------
    list de tuples representant (x, y) pour l'ensemble d'entraînement,
    de validation et l'ensemble de test
    """

    data = load_full_mnist(which_set)

    return [(x[:x.shape[0]/10], y[:y.shape[0]/10]) for x, y in data]


def load_full_mnist():
    """
    Charge la version complète de l'ensemble MNIST
    (50000 examples d'entraînement)

    Returns
    -------
    list de tuples representant (x, y) pour l'ensemble d'entraînement,
    de validation et l'ensemble de test
    """

    data = cPickle.load(gzip.open('mnist.pkl.gz', 'rb'))
    return cast_data(data)


class SGDTraining(object):
    """
    Entraînement par descente de gradient stochastique.
    (Stochastic Gradient Descent)

    La classe elle même est abstraite et doit être hérité d'un
    modèle pour faire son entraînement.

    Attributes
    ---------
    message_frequency: float
        La fréquence à laquelle le message est mis à jour 
        (Pourcentage du nombre maximal d'époque)
    epochs: list of int
        Liste des époques où il y a eu enregistrement du coût
        et de la perte du modèle
    loss_curves: list of float
        Liste de la perte du modèle à chaque mise à jour du message
    cost_curves
        Liste du coût du modèle à chaque mise à jour du message

    Methods
    -------
    train(train_data, learning_rate, max_epoch, train_labels=None, 
          batch_size=128, stopping_rule=None, monitoring_data=None)
        Entraine le modèle jusqu'à max_epoch ou jusqu'à ce que stopping_rule
        renvoie True

    """
    def __init__(self, message_frequency=0.01):
        """
        Parameters
        ----------
        message_frequency: float
            La fréquence à laquelle le message est mis à jour 
            (Pourcentage du nombre maximal d'époque)
        """

        self.message_frequency = message_frequency

    def train(self,train_data, train_labels, learning_rate, max_epoch,
              batch_size=128, stopping_rule=None, monitoring_data=None):
        """
        Entraînement le modèle. La méthode doit être hérité d'un modèle 
        pour être utilisé.

        Parameters
        ---------
        train_data: ndarray
            Matrice d'exemples de format (n,d) où n est le nombre d'exemple et 
            d la dimension
        train_labels: ndarray
            Vecteur de cibles, de format n
        learning_rate: float
            taux d'apprentissage pour l'entraînement
        max_epoch: int
            nombre maximal d'époque pendant l'entraînement
        batch_size: int, default is 128
            nombre d'exemples par mini-batch
        stopping_rule: object, default to None
            object avec méthode __call__ qui prend en paramètre train_data et 
            train_labels et renvoie True si le critère d'arrêt est atteint,
            False sinon
        monitoring_data: dict
            dictionnaire d'ensemble de données pour le monitoring, chaque
            valeur (value) doit être un tuple (data, labels)
        """

        assert hasattr(self, "update"), ("Le modèle n'a pas de méthode "
            "'update' pour mettre à jour les poids")

        assert hasattr(self, "compute_cost"), ("Le modèle n'a pas de méthode "
            "'compute_cost' pour calculer le coût sur un ensemble")

        # Engregistre la valeur du taux d'apprentissage pour l'appel de la 
        # fonction self.update (au cas où le taux d'apprentissage serait 
        # changeant d'une époque à l'autre)
        self.learning_rate.set_value(learning_rate)

        data_keys = [u"ensemble d'entraînement"]
        if monitoring_data is not None:
            data_keys += monitoring_data.keys()

        self.epochs = []
        self.loss_curves = OrderedDict((name, []) for name in data_keys)
        self.cost_curves = OrderedDict((name, []) for name in data_keys)

        last_msg_epoch = 0
    
        for epoch in xrange(max_epoch):
            for mini_batch in xrange(0, train_data.shape[0], batch_size):
                if train_labels is not None:
                    self.update(train_data[mini_batch:mini_batch+batch_size],
                                train_labels[mini_batch:mini_batch+batch_size])
                else:
                    self.update(train_data[mini_batch:mini_batch+batch_size])

            # Calcule les coûts et pertes et mets à jour le message
            if (epoch-last_msg_epoch)/float(max_epoch) > self.message_frequency:

                cost, loss, costs, losses = self._compute_monitoring(
                        train_data, train_labels, monitoring_data)

                for name in costs.keys():
                    self.loss_curves[name].append(losses[name])
                    self.cost_curves[name].append(costs[name])

                self.epochs.append(epoch)

                print "\r%3d%% : époque %d : perte = %f" % \
                    (int(100*epoch/float(max_epoch)), epoch, loss),

                last_msg_epoch = epoch

            # Vérifie que l'object est "callable" et test si le critère 
            # d'arrêt est atteint
            if hasattr(stopping_rule, "__call__") and \
               stopping_rule(train_data,train_labels):
                # change de ligne pour compenser le signe '\r'
                print "\nRègle d'arrêt atteinte"
                break

        # change de ligne pour compenser le signe '\r'
        print ""

    def _compute_monitoring(self, train_data, train_labels, monitoring_data):

        costs = {}
        losses = {}
        cost = self.compute_cost(train_data, train_labels)
        loss = self.compute_loss(train_data, train_labels)

        costs[u"ensemble d'entraînement"] = cost
        losses[u"ensemble d'entraînement"] = loss

        if monitoring_data is not None:
            for name, [data, labels] in monitoring_data.items():
                costs[name] = self.compute_cost(data, labels)
                losses[name] = self.compute_cost(data, labels)
            
        return cost, loss, costs, losses


class FeedForwardNeuralNetLayer:
    """
    Couche d'un réseau de neurones feedforward

    Attributes
    ---------
    non_linearities: dict
        dictionnaire de non-linéarités (peut être augmenté avec tanh et relu)
    non_linearity: object
        non-linéarité de la couche. La fonction est extraite du dictionnaire
        de non-linéarités
    W: theano.shared
        paramètres W de la couche, enregistré comme variable «partagée» theano
    b: theano.shard
        paramètres b de la couche, enregistré comme variable «partagée» theano
    params: list
        liste des paramètres W et b de la couche

    Methods
    -------
    fprop(state_below) 
        Applique la transformation linéaire puis la non-linéarité.
    get_l1()
        calcule le coût de norme L1 pour W
    get_l2()
        calcule le coût de norme L2 pour W
    """

    non_linearities = {
        "linear": lambda state_below: state_below, 
        "sigmoid": lambda state_below: T.nnet.sigmoid(state_below), 
        "softmax": lambda state_below: T.nnet.softmax(state_below)
    }

    def __init__(self, name, n_in, n_out, non_linearity, rng=None):
        """
        Parameters
        ----------
        name: string
            Nom de la couche. Ce nom sera présent dans le nom des 
            variables W et b et sera pratique pour le débugage du graphe
            theano
        n_in: int
            nombre d'unité en entrée
        n_out: int
            le nombre d'unité en sortie
        non_linearity: string
            le nom de la non-linéarité. Doit être présent comme clé
            dans le dictionnaire des non-linéarités (non_linearities)
        rng: numpy.random.RandomState or None
            Un objet pour échantillioner les valeurs initiales de W
        """

        assert non_linearity in self.non_linearities.keys(), \
            "La non-linéarité n'est pas supportée : %s" % str(non_linearity)

        assert isinstance(n_in, int) and n_in > 0 and \
               isinstance(n_out, int) and n_out > 0

        if rng is None:
            # Crée un générateur de nombre aléatoire avec un germe précis
            rng = np.random.RandomState([2014, 10, 26])

        irange = np.sqrt(6. / (n_in + n_out))

        W = np.asarray(rng.uniform(-irange, irange, size=(n_in, n_out)),
                       dtype=theano.config.floatX)
        b = np.asarray(np.zeros(n_out), 
                       dtype=theano.config.floatX)
        self.W = theano.shared(W, name = "%s_W" % name)
        self.b = theano.shared(b, name = "%s_b" % name)

        self.params = [self.W, self.b]

        self.non_linearity = self.non_linearities[non_linearity]

    def fprop(self, state_below):
        """
        Calcul la phase de propagation avant; transformation linéare et 
        non-linéarité.

        Parameters:
        state_below: theano.Variable
            Variable theano, peut être une entrée X où la sortie d'une
            couche précédente

        Returns
        -------
        theano.Variable
            Retourne une variable theano de format (batch_size, n_out)
        """
        return self.non_linearity(T.dot(state_below, self.W) + self.b)

    def get_l1(self):
        """
        Calcule le coût de norme L1 pour W
 
        Returns
        -------
        theano.Scalar
            Retounr un scalaire theano
        """
        return T.abs_(self.W).sum()

    def get_l2(self):
        """
        Calcule le coût de norme L2 pour W
 
        Returns
        -------
        theano.Scalar
            Retounr un scalaire theano
        """
        return T.sqr(self.W).sum()


class FeedForwardNeuralNet(SGDTraining):
    """
    Réseau de neurones de type feedforward

    Attributes
    ----------
    n_in: int 
        Nombre d'unité en entrée
    n_hids: list of int
        Nombre d'unité cachée pour chaque couche cachée
        (une liste vide donne un modèle de régression logistique)
    n_out: in
        Nombre d'unité de sortie (nombre de classes)
    non_linearities: list of int or string
        Non-linéarité des couches cachées. Tous identiques si seulement 
        défini par une string et non une liste
    l1: list of float or float
        Coût de norme L1 appliqué aux W
    l2:
        Coût de norme L2 appliqué aux W
    layers: list of object
        Couche du réseau, comprend toutes les couches cachées et la couche
        de sortie
    params: list of theano.shared
        Liste de variable «partagée» theano représentant les W et b du réseau
    rng: numpy.random.RandomState or None
        Un objet pour échantillioner les valeurs initiales de W

    Methods
    -------
    train(train_data, learning_rate, max_epoch, train_labels=None, 
          batch_size=128, stopping_rule=None, monitoring_data=None)
        Hérité de SGDTraining: Entraine le modèle jusqu'à max_epoch ou jusqu'à ce que stopping_rule
        renvoie True
    compute_predictions(test_x)
        Calcule les prédictions du modèle du l'ensemble "test_x"
    compute_loss(data, labels)
        Calcule la fonction de perte sur l'ensemble "data" avec les 
        cibles "labels"
    compute_cost(data, labels)
        Calcule l'erreur de classification sur l'ensemble "data" avec les 
        cibles "labels"
    """

    def __init__(self, n_in, n_hids, n_out, non_linearities, l1=0., l2=0., rng=None):
        """
        Parameters
        ----------
        n_in: int 
            Nombre d'unité en entrée
        n_hids: list of int
            Nombre d'unité cachée pour chaque couche cachée
            (une liste vide donne un modèle de régression logistique)
        n_out: in
            Nombre d'unité de sortie (nombre de classes)
        non_linearities: list of int or string
            Non-linéarité des couches cachées. Tous identiques si seulement 
            défini par une string et non une liste
        l1: list of float or float
            Coût de norme L1 appliqué aux W
        l2:
            Coût de norme L2 appliqué aux W
        rng: numpy.random.RandomState or None
            Un objet pour échantillioner les valeurs initiales de W
        """

        super(FeedForwardNeuralNet, self).__init__()

        if rng is None:
            # Crée un générateur de nombre aléatoire avec un germe précis
            self.rng = np.random.RandomState([2014, 10, 26])

        if isinstance(non_linearities, str):
            non_linearities = [non_linearities] * len(n_hids)

        assert len(non_linearities) == len(n_hids), \
            ("Nombre de non-linéarité inégale au nombre de couches cachées : "
             "%d vs %d" % (len(non_linearities), len(n_hids)))

        if isinstance(l1, float):
            l1 = [l1] * (len(n_hids) + 1)
        if isinstance(l2, float):
            l2 = [l2] * (len(n_hids) + 1)

        assert len(l1) == len(n_hids) + 1
        assert len(l2) == len(n_hids) + 1
        for l in l1:
            assert isinstance(l, float) and l >= 0, "l1 < 0 n'a pas de sens!"
        for l in l2:
            assert isinstance(l, float) and l >= 0, "l2 < 0 n'a pas de sens!"

        self.layers = []

        # Crée chacune des couches du réseaux
        for i, [layer_n_in, layer_n_out, layer_nonlin] in \
                enumerate(itertools.izip([n_in] + n_hids, n_hids, 
                                         non_linearities)):
            layer = FeedForwardNeuralNetLayer("h_%d" % i, layer_n_in, layer_n_out, 
                                              layer_nonlin, self.rng)

            self.layers.append(layer)

        # Crée la dernière couche qui est toujours présente peut importe les 
        # couches cachées
        last_layer_in = n_hids[-1] if n_hids else n_in
        last_layer = FeedForwardNeuralNetLayer("y", last_layer_in, n_out, "softmax", rng)
        self.layers.append(last_layer)

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_out = n_out
        self.rng = rng
        self.non_linearities = non_linearities
        self.l1 = l1
        self.l2 = l2
        
        self._build_theano_graph()

    @property
    def params(self):
        """ 
        Liste de variable «partagée» theano représentant les W et b du réseau
        """

        return sum((layer.params for layer in self.layers), [])

    def _build_theano_graph(self):
        """
        Construit le modèle et compile les fonctions
        """
        
        X = T.matrix()
        y = T.ivector()

        # La valeur 0 est donné pour s'assurer qu'elle sera changé à 
        # l'entraînement. Si elle n'est pas changé, ça ne fonctionnera pas.
        self.learning_rate = theano.shared(
                np.cast[theano.config.floatX](0.), 
                name="learning_rate")

        state_below = X

        for layer in self.layers:
            state_below = layer.fprop(state_below)

        p_y_given_x = state_below

        y_pred = T.argmax(p_y_given_x, axis=1)

        loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

        # on itère sur toutes les couches pour calculer le coût pour 
        # tous les W
        for l1, layer in zip(self.l1, self.layers):
            if l1 > 0.:
                loss += l1*layer.get_l1()

        for l2, layer in zip(self.l2, self.layers):
            if l2 > 0.:
                loss += l2*layer.get_l2()

        # Le gradient est calculé par rapport à tout les paramètres
        # grads est une liste contenant le gradient par rapport à 
        # chaque paramètre, dans le même ordre que dans self.params
        grads = T.grad(loss, self.params)

        updates = OrderedDict()

        # On itère sur self.params et grads en même temps 
        # Les paramètres sont tous mis à jour de la même façon
        for param, grad in itertools.izip(self.params, grads):
            updates[param] = param - self.learning_rate * grad

        # La fonction update fera la propagation avant puis la 
        # rétro-propagation pour mettre à jour les paramètres
        self.update = function([X, y], y_pred, updates=updates)

        # une fonction qui renvoie les prédictions du modèle",
        self.predict = function([X], y_pred)

        # une fonction qui renvoie le taux d'erreur du modèle",
        self.compute_error_rate = function([X, y], T.mean(T.neq(y_pred, y)))

        # une fonction qui renvoie le résultat de la fonction de perte
        self.compute_loss = function([X, y], loss)

    def compute_predictions(self, test_x):
        """
        Calcule les prédictions du modèle du l'ensemble "test_x"
        """
        return self.predict(test_x)

    def compute_loss(self, data, labels):
        """
        Calcule la fonction de perte sur l'ensemble "data" avec les 
        """
        return float(self.compute_loss(data, labels))

    def compute_cost(self, data, labels): 
        """
        Calcule l'erreur de classification sur l'ensemble "data" avec les 
        """
        return float(self.compute_error_rate(data, labels))


class MultiClassLogisticRegression(FeedForwardNeuralNet):
    """
    Modèle de régression logistique pour faire de la classification multiclasse

    Attributes
    ----------
    n_in: int 
        Nombre d'unité en entrée
    n_out: in
        Nombre d'unité de sortie (nombre de classes)
    l1: list of float or float
        Coût de norme L1 appliqué aux W
    l2:
        Coût de norme L2 appliqué aux W
    params: list of theano.shared
        Liste de variable «partagée» theano représentant les W et b du réseau
    rng: numpy.random.RandomState or None
        Un objet pour échantillioner les valeurs initiales de W

    Methods
    -------
    train(train_data, learning_rate, max_epoch, train_labels=None, 
          batch_size=128, stopping_rule=None, monitoring_data=None)
        Hérité de SGDTraining: Entraine le modèle jusqu'à max_epoch ou jusqu'à ce que stopping_rule
        renvoie True
    compute_predictions(test_x)
        Calcule les prédictions du modèle du l'ensemble "test_x"
    compute_loss(data, labels)
        Calcule la fonction de perte sur l'ensemble "data" avec les 
        cibles "labels"
    compute_cost(data, labels)
        Calcule l'erreur de classification sur l'ensemble "data" avec les 
        cibles "labels"
    """

    def __init__(self, n_in, n_out, l1=0., l2=0., rng=None):
        """
        Parameters
        ----------
        n_in: int 
            Nombre d'unité en entrée
        n_out: in
            Nombre d'unité de sortie (nombre de classes)
        l1: list of float or float
            Coût de norme L1 appliqué aux W
        l2:
            Coût de norme L2 appliqué aux W
        rng: numpy.random.RandomState or None
            Un objet pour échantillioner les valeurs initiales de W
        """

        super(MultiClassLogisticRegression, self).__init__(n_in, [], n_out, 
                                                           [], l1, l2, rng)



def plot_training_curves(epochs, learning_curves, title, ylabel, 
                         xlabel=u"Époques", xlog=False, ylog=False):
    """
    Parameters
    ----------
    epochs: list
        Liste représentant les époques
    learning_curves: dict
        Dictionnaire de courbe. Chaque clé sera utilisé pour identifier la 
        courbe dans la légende
    title: string
        Titre du graphique
    ylabel: string
        Nom de l'axe y
    xlabel: string
        Nom de l'axe x
    xlog: bool, default to False
        L'axe x est affiché sous format logarithmique si True
    ylog: bool, default to False
        L'axe y est affiché sous format logarithmique si True
    """

    figure = plt.figure()#figsize=(8,6))
    axes = plt.subplot(111)

    handlers = []

    for name, curve in learning_curves.items():
        handler = axes.plot(epochs, curve)[0]

        handlers.append(handler)
    
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    if xlog:
        axes.set_xscale('log')
    if ylog:
        axes.set_yscale('log')

    figure.legend(handlers, learning_curves.keys(), loc="right")
    plt.title(title)

    plt.show()


def plot_decision_frontiers(classifieur, train_data, train_labels, test_data, test_labels, n_points=50):

    train_test = np.vstack((train_data,test_data))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = np.linspace(min_x1,max_x1,num=n_points).astype(theano.config.floatX)
    ygrid = np.linspace(min_x2,max_x2,num=n_points).astype(theano.config.floatX)

	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = np.array(combine(xgrid,ygrid))

    les_sorties = classifieur.compute_predictions(thegrid)
    if les_sorties.ndim == 1:
        classesPred = np.sign(les_sorties - 0.5)
    else:
        classesPred = np.argmax(les_sorties, axis=1)

    # La grille
    # Pour que la grille soit plus jolie
    #props = dict( alpha=0.3, edgecolors='none' )
    plt.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, s=50)
    # Les points d'entrainment
    plt.scatter(train_data[:,0], train_data[:,1], c = train_labels, marker = 'v', s=50)
    # Les points de test
    plt.scatter(test_data[:,0], test_data[:,1], c = test_labels, marker = 's', s=50)

    ## Un petit hack, parce que la fonctionalite manque a pylab...
    h1 = plt.plot([min_x1], [min_x2], marker='o', c = 'w',ms=5) 
    h2 = plt.plot([min_x1], [min_x2], marker='v', c = 'w',ms=5) 
    h3 = plt.plot([min_x1], [min_x2], marker='s', c = 'w',ms=5) 
    handles = [h1,h2,h3]
    ## fin du hack

#    labels = ['grille','train','test']
#    plt.legend(handles,labels)

    plt.axis('equal')
    plt.show()


## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout



