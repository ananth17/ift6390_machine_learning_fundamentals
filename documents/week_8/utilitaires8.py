# -*- coding:utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

import time

colors = [(31, 119, 180),  
          (255, 127, 14), 
          (152, 223, 138), 
          (214, 39, 40), 
          (148, 103, 189), 
          (197, 176, 213), 
          (140, 86, 75), 
          (227, 119, 194), 
          (127, 127, 127),
          (188, 189, 34), 
          (23, 190, 207), 
          (158, 218, 229)]


for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r / 255., g / 255., b / 255.)


def generate_data(K, N=100, mean=0., std=10., rng=None):
    """
    Generateur aleatoire de donnee.

    K moyennes sont genere selon une distribution normale
    de moyenne `mean` et de variance `std`. Les donnees
    sont ensuite genere au hasard autour des K moyennes
    avec une variance de 2.

    Arguments
    ---------
    K: int
        nombre de gaussiennes genere
    N: int
        nombre d'exemples generes par gaussiennes
    mean: float, default=0
        moyenne pour generer le centre des gaussiennes
    std: float, default=10
        variance pour generer le centre des gaussiennes
    rng: numpy.random.RandomState ou None 
        objet pour echantillonner les donnees

    Returns
    -------
    ndarray
        matrice de format (K*N, 2)
    """
    if rng is None:
        rng = np.random.RandomState()

    mu = rng.normal(mean, std, size=(K, 2))

    X = rng.normal(mu.flatten(), 2., size=(N, K*2))
    X = X.reshape((N*K, 2))

    return X

class DummyModel:
    def __init__(self):
        pass

class Tests:
    def __init__(self):
        self.data = np.random.RandomState(1).normal(size=(5, 2))
        self.model = DummyModel()
        self.model.k = 3
        self.model.rng = np.random.RandomState(2)
        z = [0, 1, 2, 0, 1]
        self.model.mu = self.data[:self.model.k]
        self.z = np.zeros((self.data.shape[0], self.model.k))
        self.z[np.arange(self.data.shape[0]), z] = 1.
        self.cost = 1.43292789524

    def test_init_centroids_naif(self, init_fct):
        self.model.mu = None
        init_fct(self.model, self.data)
        assert self.model.mu.shape[0] == self.model.k, \
                "Le nombre de centroide K n'est pas respecte : %d" \
                    % self.model.mu.shape[0]
        assert self.model.mu.shape[1] == 2, \
                ("Les centroides n'ont pas le meme nombre de dimension "
                 "que les donnees : %d") % self.model.mu.shape[1]
        assert np.all(self.model.mu.min(0) >= self.data.min(0)), \
                ("Une des dimensions d'un centroide est plus petite "
                 "que les donnees")
        assert np.all(self.model.mu.max(0) <= self.data.max(0)), \
                ("Une des dimensions d'un centroide est plus grand "
                 "que les donnees")

        print "OK! :D"

    def test_z(self, z_fct):
        z = z_fct(self.data, self.model.mu)
        
        if not np.all(self.z == z):
            print "Mauvais resultat, voici la difference"
            print z - self.z
            assert np.all(self.z == z), "Le resultat n'est pas bon" 
        
        print "OK! :D"

    def test_mise_a_jour(self, up_fct):
        up_fct(self.model, self.data, self.z)
        
        mu = np.array([[ 1.68457856, -0.68648166],
                       [-0.10456633, -0.6611695 ],
                       [ 0.86540763, -2.3015387 ]])

        assert self.model.mu.shape[0] == mu.shape[0], \
                "le nouveau mu n'a pas les bonnes dimensions"
        assert self.model.mu.shape[1] == mu.shape[1], \
                "le nouveau mu n'a pas les bonnes dimensions"
        
        if not np.allclose(mu, self.model.mu):
            print "Mauvais resultat, voici la difference"
            print self.model.mu - mu
            assert np.allclose(mu, self.model.mu), \
                    "Le resultat n'est pas bon" 

        print "OK! :D"

    def test_cout(self, cost_fct):
        cost = cost_fct(self.model.mu, self.data, self.z)

        assert np.allclose(cost, self.cost), "Mauvais resultats"

        print "OK! :D"

    def time_z(self, init_fct, fct1, fct2, tests=50):

        data = np.random.RandomState(1).normal(size=(5000, 2))
        init_fct(self.model, data)
        self.time("Z naif", fct1, "Z rapide", fct2, tests,
                  data, self.model.mu)

    def time_mise_a_jour(self, init_fct, z_fct, fct1, fct2, tests=50):

        data = np.random.RandomState(1).normal(size=(10000, 2))
        init_fct(self.model, data)
        z = z_fct(data, self.model.mu)
        self.time("Mise a jour naive", fct1, "Mise a jour rapide", fct2, tests,
                  self.model, data, z)

    def time_cout(self, init_fct, z_fct, fct1, fct2, tests=50):

        data = np.random.RandomState(1).normal(size=(10000, 2))
        init_fct(self.model, data)
        z = z_fct(data, self.model.mu)
        self.time("Coût naif", fct1, "Coût rapide", fct2, tests,
                  self.model.mu, data, z)

    def time(self, name1, fct1, name2, fct2, tests, *params):

        times = []

        start = time.clock()
        for test in xrange(tests):
            fct1(*params)
        elapsed = (time.clock() - start)/float(tests)

        print "%s a pris %f secondes" % (name1, elapsed)

        times = []


        start = time.clock()
        for test in xrange(tests):
            fct2(*params)
        elapsed = (time.clock() - start)/float(tests)

        print "%s a pris %f secondes" % (name2, elapsed)


def plot_costs_and_clusterings(iteration, iterations, costs, mu, z, data, 
                               title, dmu=None):
    """
    Affiche un graphique de la courbe d'apprentissage et un graphique des k-means

    Attributes
    ----------
    iteration: int
        iteration courante (pas incluse dans iterations pendant la premiere phase E)
        car le coût n'est pas encore calcule
    iterations: list de int
        chaque iteration (axe x) correspondant au coût (axe y)
    costs: list de float
        les coûts a chaque iteration monitore
    mu: ndarray
        Parametres du k-means. matrice de dimension (k, d) ou k est le nombre 
        de centroide et d est le nombre de dimension des donnees
    z: ndarray
        Variables latentes. matrice de dimension (n, k) ou n est le nombre de donnees
        et k est le nombre de centroides.
    data: ndarray
        matrice de donnees de dimension (n, d) ou n est le nombre d'exemples
        et d est le nombre de dimensions
    title: string
        titre principal du graphique
    dmu: ndarray ou None, default=None
        derniers parametres du modele avant la mise a jour. Sert a dessiner 
        les fleches montrant le mouvement des centroides.
    """

    figure = plt.figure(figsize=(15,5))
    plt.suptitle(title, fontsize=20)
    plot_costs(iterations, costs, 121)
    plot_clusterings(mu, z, data, 122, 
        u"%d iteration%s" % (iteration, "s" if iteration>2 else ""), dmu)
    plt.show()

def plot_costs(iterations, costs, subplot=None):
    """
    Affiche un graphique de la courbe d'apprentissage des k-means

    Attributes
    ----------
    iterations: list de int
        chaque iteration (axe x) correspondant au coût (axe y)
    costs: list de float
        les coûts a chaque iteration monitore
    subplot: int ou None, default=None
        sert a definir la position s'il est un subplot
    """

    if subplot is None:
        figure = plt.figure()
        axes = plt.subplot(111)
    else:
        axes = plt.subplot(subplot)

    if len(iterations) > 2:
        axes.plot(iterations, costs)[0]

        axes.set_xlabel(u"Iterations")
        axes.set_ylabel(u"Coût J")

        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        axes.get_xaxis().tick_bottom()
        axes.get_xaxis().set_ticks(iterations)
        y = np.linspace(min(costs), max(costs),5)
        axes.get_yaxis().tick_left()
        axes.get_yaxis().set_ticks(y)
        axes.get_yaxis().set_ticklabels(["%dk" % (c/1000.) for c in y])

        plt.title(u"Courbe d'apprentissage")
    else:
        plt.axis("off")
        plt.title(u"Moins de deux iterations")

    if subplot is None:
        plt.show()


def plot_clusterings(mu, z, data, subplot=None, title=None, dmu=None):
    """
    Affiche un graphique des k-means avec les donnees

    Attributes
    ----------
    mu: ndarray
        Parametres du k-means. matrice de dimension (k, d) ou k est le nombre 
        de centroide et d est le nombre de dimension des donnees
    z: ndarray
        Variables latentes. matrice de dimension (n, k) ou n est le nombre de donnees
        et k est le nombre de centroides.
    data: ndarray
        matrice de donnees de dimension (n, d) ou n est le nombre d'exemples
        et d est le nombre de dimensions
    subplot: int ou None, default=None
        sert a definir la position s'il est un subplot (sous-graphique)
    title: string
        titre du (sous-)graphique
    dmu: ndarray ou None, default=None
        derniers parametres du modele avant la mise a jour. Sert a dessiner 
        les fleches montrant le mouvement des centroides.
    """

    if subplot is None:
        axes = plt.subplot(111)
    else:
        axes = plt.subplot(subplot)


    for cluster in xrange(z.shape[1]):
        cluster_data = data[np.where(z[:,cluster])]
        axes.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster % len(colors)],
                    alpha=0.25)
        if dmu is not None:
            X = dmu[cluster, 0]
            Y = dmu[cluster, 1]
            U = (mu[cluster, 0] - X)
            V = (mu[cluster, 1] - Y)
            axes.quiver(X, Y, U, V, linewidths=0.01, scale=1, angles="xy", scale_units="xy")
        axes.scatter(mu[cluster, 0], mu[cluster, 1], marker='*', 
                    edgecolor='black', color=colors[cluster % len(colors)], 
                    linewidths=2, s=200, alpha=0.75)

    plt.axis('off')

    if title:
        plt.title(title)

    if subplot is None:
        plt.show()
