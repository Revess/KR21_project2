from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import copy
import numpy as np

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def pruneNetwork(self, evidence=dict()):
        cpts = self.bn.get_all_cpts()
        # remove edges
        for node in cpts.keys():
            for ev in [ev for ev in evidence if ev in cpts[node].keys() and ev != list(cpts[node].keys())[-2]]: # Make sure the evidence is in the node and that the node is not the evidence itself
                new_cpt = cpts[node][cpts[node][ev] == evidence[ev]]
                new_cpt = new_cpt.drop(ev,axis=1)
                self.bn.update_cpt(node, new_cpt)
                # Delete the edge itself
                self.bn.del_edge((ev,list(cpts[node].keys())[-2]))
                # Now also remove the node
                if len(self.bn.get_children(node)) == 0 and len(new_cpt[node].keys()) == 2 and len(self.bn.get_parents(node)) == 0:
                    self.bn.del_var(node)
                if len(self.bn.get_children(ev)) == 0 and len(self.bn.get_parents(ev)) == 0:
                    self.bn.del_var(ev)

    def reduceNet(self, evidence=dict()):
        cpts = self.bn.get_all_cpts()
        for node in cpts.keys():
            if sum([1 if ev in cpts[node].keys() else 0 for ev in evidence]) >= 1:
                newCPT = self.bn.reduce_factor(pd.Series(evidence),cpts[node])
                newCPT = newCPT[newCPT.p != 0]
                self.bn.update_cpt(node, newCPT)          

    def maxingOut(self, variable, cpt):
        if len([col for col in cpt.columns if 'ins. of' in col]) > 1:
            print('Weird amount of instances columns, please check')
            return

        df = cpt
        res = pd.DataFrame(columns=df.columns.drop([variable]))

        sort = list(df.columns)
        sort.remove('p')
        sort.remove(variable)

        len_ins = len([col for col in df.columns if 'ins. of' in col])
        if len_ins == 1:
            sort.remove('ins. of')

        df = df.sort_values(by = sort, ignore_index=True)

        for i in range(len(df.iloc[:,0])):
            if i % 2 == 0:
                max = df.loc[i:i, ['p', variable]]
            else:
                if df.loc[i, 'p'] > max.iloc[0, 0]:
                    max = df.loc[i:i, ['p', variable]]
                maxres = df.drop([variable, 'p'], axis=1).loc[i:i, :]
                maxres['p'] = max.iloc[0, 0]
                if len_ins == 0:
                    maxres['ins. of'] = variable + ': ' + str(max.iloc[0, 1])
                else:
                    maxres['ins. of'] += ', ' + variable + ': ' + str(max.iloc[0, 1])
                res = pd.concat([res, maxres], axis=0, sort=False, ignore_index=True)

        return res

    def factorMultiplication(self, factor1, factor2):
        cpts = self.bn.get_all_cpts()
        X = cpts[factor1]
        Z = cpts[factor2]
        union = list(set(X.columns).intersection(Z.columns))
        union.remove('p')
        cols = list(pd.concat([X, Z]).columns)
        cols.remove('p')
        res = pd.DataFrame(columns=cols + ['p'])
        for x in range(len(X.iloc[:, 0])):
            for z in range(len(Z.iloc[:, 0])):
                if X.loc[x, union[0]] == Z.loc[z, union[0]]:
                    mul = X.loc[x, 'p'] * Z.loc[z, 'p']
                    df = pd.merge(X.loc[x:x, X.columns != 'p'], Z.loc[z:z, Z.columns != 'p'])
                    df['p'] = mul
                    res = pd.concat([res, df])
        return res

    def Ordering(self, heuristic):
        if heuristic == 'min-degree':
            degrees = dict(self.bn.get_interaction_graph().degree)
            graph = copy.deepcopy(self.bn.get_interaction_graph().adj)
            order = []
            for i in range(len(degrees)):
                e = min(degrees, key=degrees.get)
                order += [e]
                new_edges = []
                for j in graph:
                    if e in graph[j]:
                        if j in degrees:
                            degrees[j] -= 1
                            new_edges += [j]
                for ne in range(len(new_edges) - 1):
                    for ae in range(ne + 1, len(new_edges)):
                        if new_edges[ae] not in graph[new_edges[ne]]:
                            print('adding', ne)
                            degrees[ne] += 1
                del degrees[e]
            return order
        elif heuristic == 'min-fill':
            graph = copy.deepcopy(self.bn.get_interaction_graph().adj)
            nodes = list(self.bn.get_interaction_graph().nodes)
            order = []
            for i in range(len(nodes)):
                minimal = np.inf
                for n in nodes:
                    n_edges = 0
                    new_edges = []
                    for g in graph:
                        if n in graph[g]:
                            new_edges += [g]
                    for ne in range(len(new_edges) - 1):
                        for ae in range(ne + 1, len(new_edges)):
                            if new_edges[ae] not in graph[new_edges[ne]]:
                                n_edges += 1
                    if minimal > n_edges:
                        minimal = n_edges
                        add = n
                order += [add]
                nodes.remove(add)
            return order
        else:
            print('wrong heuristic chosen, pick either min-degree or min-fill')

    def variableElimination(self, query, evidence=dict()):
        print(evidence)
        cpts = self.bn.get_all_cpts()
        instantiation = pd.Series(evidence)
        self.bn.draw_structure()
        print(cpts)

        order = reasoner.Ordering('min-degree')

        for i in order:
            if i not in query:
                if i not in evidence.keys():
                    result = self.marginalization(1, reasoner.bn.get_all_cpts()[i])
                    self.bn.update_cpt(i,result)
        print()

    def marginalDistributions(self, query, evidence=dict()):
        cpts = self.bn.get_all_cpts()


        reducing = self.reduceNet(evidence)
        
        graph = self.bn.get_interaction_graph()
        nx.all_simple_paths(graph,source=X[0],target=Y[0])

        Qande = []

        for node in cpts.keys():
            for q in query:
                if q in cpts.iloc[0]:
                    Qande.append(node)

        for node in cpts.keys():
            for e in evidence.keys():
                if e in cpts.iloc[0]:
                    Qande.append(node)
        
        print(Qande)

        neededcpts = {}

        for item in Qande:
            neededcpts[item] = self.bn.get_cpt(item)

        #have to use order with the only the cpts I got, can I give them to this function?

        for q in query:
            result = self.marginalization(q, )

    def mapping(self, query, evidence=dict()):
        
        order = self.Ordering('min-degree')
        
        print('begin order', order)

        order = [x for x in order if x not in query]

        print('later order',order)

        for i in order:
            result = self.marginalization(i, reasoner.bn.get_all_cpts()[i])
            self.bn.update_cpt(i, result)

        for q in query:
            domax = self.maxingOut(q)
            self.bn.update_cpt(q, domax)

            print(domax.loc[:,'p'])
            print(domax.loc[:,'ins. of ' + q])

            print(domax['p'])
            print(domax.loc[:,'ins. of' + q])


        #compute P(Q,e) first with variable elimination, then maximize-out Q using extended variables

    def mpe(self, query, evidence=dict()):
        order = self.Ordering('min-degree')
        ending = []

        for i in order:
            if i not in query:
                if i not in evidence.keys():
                    result = self.maxingOut(i)
                    self.bn.update_cpt(i, result)
                    print('this is the cpt:', result)
                    ending.append(result)

        print('what is the query', query)   

        for q in query:

            print(result['p']) #this should work, right? print the p value of the query cpts
            print(result['ins. of ' + q])
        #return query['p']
        print('this is the ending cpt:', ending)


    def dSeperation(self, X=list(), Y=list(), Z=list()):
        graph = self.bn.get_interaction_graph()
        [graph.remove_node(z) for z in Z]
        try:
            return (False if len(list(nx.all_simple_paths(graph,source=X[0],target=Y[0]))) == 0 else True)
        except:
            return False

    def independence(self, X=list(), Y=list(), Z=list()):
        return not self.dSeperation(X,Y,Z)

    def marginalization(self, X, cpt):
        return cpt.groupby([value for value in list(cpt.columns) if value in self.bn.get_all_variables()])['p'].sum().reset_index()

reasoner = BNReasoner("./testing/dog_problem.BIFXML")
x = reasoner.maxingOut(variable='dog-out',cpt = reasoner.bn.get_all_cpts()['dog-out'])

