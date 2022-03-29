# import packages
import dimod

import neal
from pyqubo import Spin, Array, Placeholder

import numpy as np
import igraph as ig

import os
import pickle


sampler = neal.SimulatedAnnealingSampler()

#############
# Define Class
#############


class GraphPartitioning(ig.Graph):
    '''
    A child class of the igraph.Graph class with added methods for the graph partitioning algorithm.
    '''
    # inherite the init function of ig.Graph and add some attributes right away.

    def __init__(self, *args, **kwds):
        super().__init__(self, *args, **kwds)
        # vertice attributes
        for v in self.vs():
            v['spin'] = -1
            v['label'] = v.index
            v['color'] = 'red'
        # edge attributes
        for e in self.es():
            e['connecting'] = False
            e['width'] = 4
            e['color'] = 'black'
        # counting
        self.n = self.vcount()
        self.deg = max(self.degree())
        ###################
        # make PyQUBO model
        ###################
        # create the spin variables
        spin = Array.create('s', shape=self.n, vartype='SPIN')
        # so that these can be changed after compiling the model
        A = Placeholder('A')
        B = Placeholder('B')
        # create the hamiltonian according to formulas
        H_A = np.sum(spin)**2
        H_B = np.sum([(1 - spin[i]*spin[j]) for (i, j) in self.get_edgelist()])
        H = A*H_A + B*H_B
        self.model = H.compile()

    def update(self, sample_list: iter):
        '''update the spin values of the graph according to a given iterable of spin values'''
        # updating spins
        for v in self.vs():
            v['spin'] = sample_list[v.index]
            if v['spin'] == +1:
                v['color'] = 'green'
            else:
                v['color'] = 'red'
        # updating edges
        for e in self.es():
            sourcespin = self.vs.find(e.source)['spin']
            targetspin = self.vs.find(e.target)['spin']
            connecting = (sourcespin*targetspin == -1)
            e['connecting'] = connecting
            if connecting:
                e['width'] = 2
                e['color'] = 'red'
            else:
                e['width'] = 4
                e['color'] = 'black'

    def reset(self):
        '''reset the spin values to initial conditions.
        For Testing purposes'''
        self.update(np.full(shape=self.vcount(), fill_value=-1))

    def sample(self, sa=sampler, num_reads=100, show=True, **kwargs):
        """
        Generates the bqm and samples with globally defined sampler
        **kwargs:
            goalConstant: Constant factor infront of the goalfunction,
                standard value 1
            difference_func: function(n,deg), some function dependent on verticenumber(problemsize) and the degree of the graph
                standard value is constant 10
            conditionPenalty: Constant factor infront of the condition
                standard value is np.ceil(goalConstant/4 * min(2*deg, n)) + difference_func(n,deg)
        Output:
            self.sempleset: entire response from the sampler
            self.bestsampleset: the samples with lowest energy (sampleset.lowest())
            self.solution_graphs: List containing one graph object for every sample with lowest energy
        """
        # extract the kwargs and calculate standard values
        goalConstant = kwargs.get('goalConstant', 1)
        difference_func = kwargs.get('difference_func', lambda n, deg: 10)
        conditionPenalty = kwargs.get(
            "conditionPenalty", np.ceil(
                goalConstant/4 * min(2*self.deg, self.n)) + difference_func(self.n, self.deg)
        )
        # make bqm and sample
        bqm = self.model.to_bqm(
            feed_dict={'A': conditionPenalty, 'B': goalConstant})
        # I think the SimulatedAnealiner can only handle binary valued models
        self.sampleset = sa.sample(
            bqm, num_reads=num_reads).change_vartype(dimod.SPIN)
        # evaluate the result
        # acces samples by self.best_sampleset.record['sample']
        self.best_sampleset = self.sampleset.lowest()
        self.solution_graphs = []
        for s in self.best_sampleset.record['sample']:
            new_graph = self.copy()
            new_graph.update(s)
            self.solution_graphs.append(new_graph)
        # print the sample if prompted
        if show:
            print(self.sampleset)

    def save(self, name='new_example'):
        folder = f"graph_examples\\{name}"
        if not(os.path.isdir(folder)):
            os.makedirs(folder)

        with open(f'{folder}\\sampleset.pkl', 'wb') as outfile:
            pickle.dump(self.sampleset, outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        # read with:
        # with open('file', 'rb') as f:
        #   return pickle.load(f)
        with open(f'{folder}\\result.txt', 'w') as outfile:
            outfile.write(f'{self.sampleset}')

        with open(f'{folder}\\info.txt', 'w') as outfile:
            outfile.write(f'{self.sampleset.info}')

        with open(f'{folder}\\adjacency.txt', 'w') as outfile:
            outfile.writelines(str(self.get_adjacency()))

        pathnum = 1
        for g in self.solution_graphs:
            path = f'{folder}\\partition{pathnum}.png'
            ig.plot(g, target=path)
            pathnum += 1

    def check_condition(self):
        return (np.sum(self.best_sampleset.record[0]['sample']))/2

    def analyse(self, sa=sampler, num_reads=100, show=False, name='new_example', **kwargs):
        """
        Generates the bqm, samples sampler sa (default is neal.SimulatedAnnealingSampler()) and saves the results
        sa: sampler to be used
        num_reads: number of iterations
        show: wether to show the results of the sampling
        **kwargs:
            goalConstant: Constant factor infront of the goalfunction,
                standard value 1
            difference_func: function(n,deg), some function dependent on verticenumber(problemsize) and the degree of the graph
                standard value is constant 10
            conditionPenalty: Constant factor infront of the condition
                standard value is np.ceil(goalConstant/4 * min(2*deg, n)) + difference_func(n,deg)
        Output:
            self.sempleset: entire response from the sampler
            self.bestsampleset: the samples with lowest energy (sampleset.lowest())
            self.solution_graphs: List containing one graph object for every sample with lowest energy
        """
        self.sample(sa=sa, num_reads=num_reads, show=show, **kwargs)
        print(f'condtiton failed by: {self.check_condition()}')
        self.save(name=name)
