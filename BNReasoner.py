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

    def pruneNetwork(self, Q=list(), evidence=dict()): #leaf nodes die in Q voorkomen mogen toch ook niet gedelete worden? Moet hier geen rekening mee gehouden worden
        # Q is list van values.
        cpts = self.bn.get_all_cpts()
        # remove edges
        for node in cpts.keys():
            for ev in [ev for ev in evidence if ev in cpts[node].keys() and ev != list(cpts[node].keys())[-2]]: # Make sure the evidence is in the node and that the node is not the evidence itself
                if node not in Q:
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

    def factorMultiplication(self, f1=pd.DataFrame(), f2=pd.DataFrame()):
        equalColumns = [name for name in f1.columns if name in list(f2.columns) and name != "ins. of" and name != "p" ]
        f1 = f1.sort_values(equalColumns)
        f2 = f2.sort_values(equalColumns)
        Tfunc = f1.merge(f2, on=equalColumns, how="outer")
        Tfunc['p'] = Tfunc['p_x'] * Tfunc['p_y']
        Tfunc = Tfunc.drop(["p_x","p_y"], axis=1)

        if "ins. of_x" in Tfunc.columns and "ins. of_y" in Tfunc.columns:
            Tfunc["ins. of"] = [f"{x} {y}" for x, y in zip(Tfunc["ins. of_x"], Tfunc["ins. of_y"])]
            Tfunc = Tfunc.drop(["ins. of_x", "ins. of_y"], axis=1)

        if 'ins. of' in Tfunc.columns:
            Tfunc = Tfunc[list(Tfunc.drop('ins. of', axis=1).columns) + ['ins. of']] #Change the order of the columns (with its values)
        return Tfunc

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

    def variableElimination(self, query=list(), evidence=dict()):
        self.reduceNet(evidence=evidence) # First we set our evidence to True
        cpts = self.bn.get_all_cpts()
        order = [var for var in self.Ordering('min-degree') if var not in query]
        func = []
        for var in order:
            funcKeys = [key for key, cpt in copy.deepcopy(cpts).items() if var in cpt.columns]
            func = [cpts.pop(key) for key, cpt in copy.deepcopy(cpts).items() if var in cpt.columns] ##SUM-OUT over the values
            while len(func) > 1:
                f1,f2 = copy.deepcopy(func[0]), copy.deepcopy(func[1])
                del func[1]
                func[0] = self.factorMultiplication(f1,f2)
            func = [cpt[[name for name in cpt.columns if name != var and name != 'p']+ [var] + ['p']] for cpt in func]
            func = [self.marginalization(cpt) if cpt.columns[-2] not in query + list(evidence.keys()) else cpt for cpt in func ]
            func = [cpt.drop(cpt.columns[-2],axis=1) if cpt.columns[-2] not in query + list(evidence.keys()) else cpt for cpt in func]
            for index, key in enumerate(funcKeys[:len(func)]):
                cpts[key] = func[index]
        return cpts

    def marginalDistributions(self, query=list(), evidence=list()):
        self.pruneNetwork(Q=query, evidence=evidence)
        factors = self.variableElimination(query=query, evidence=evidence)
        factors['p'] = factors['p'] / factors['p'].sum()
        return factors

    def map(self, query=list(), evidence=dict(), pruning=False):
        if pruning:
            self.pruneNetwork(Q=query, evidence=evidence)
        cpts = self.variableElimination(query=query,evidence=evidence)
        func = [cpt for key, cpt in cpts.items() if sum([1 if x in query else 0 for x in cpt.columns ]) > 0]
        while len(func) > 1:
            f1,f2 = copy.deepcopy(func[0]), copy.deepcopy(func[1])
            del func[1]
            func[0] = self.factorMultiplication(f1,f2)
        func = func[0]
        for q in query:
            func = self.maxingOut(variable=q, cpt=func)
        return func

    def mpe(self, evidence=dict(), pruning=True):
        if pruning:
            self.pruneNetwork(evidence=evidence) 
        self.reduceNet(evidence=evidence)
        cpts = self.bn.get_all_cpts()
        order = self.Ordering('min-degree')
        func = []
        for var in order:
            funcKeys = [key for key, cpt in copy.deepcopy(cpts).items() if var in cpt.columns]
            func = [cpts.pop(key) for key, cpt in copy.deepcopy(cpts).items() if var in cpt.columns] ##SUM-OUT over the values
            while len(func) > 1:
                f1,f2 = copy.deepcopy(func[0]), copy.deepcopy(func[1])
                del func[1]
                func[0] = self.factorMultiplication(f1,f2)
            func = func[0]
            func = self.maxingOut(variable=var, cpt=func)
            cpts[funcKeys[0]] = func
        cpt = [x for k,x in cpts.items()][0]
        print(cpt['ins. of'])
        return cpt['p'].to_list()[0], cpt["ins. of"].to_list()[0]

    def dSeperation(self, X=list(), Y=list(), Z=list()):
        graph = self.bn.get_interaction_graph()
        [graph.remove_node(z) for z in Z]
        try:
            return (False if len(list(nx.all_simple_paths(graph,source=X[0],target=Y[0]))) == 0 else True)
        except:
            return False

    def independence(self, X=list(), Y=list(), Z=list()):
        return not self.dSeperation(X,Y,Z)

    def marginalization(self, cpt=pd.DataFrame()):
        return cpt.sort_values(list(cpt.columns[:-1])).groupby(cpt.index // 2).sum().replace(dict(zip(cpt.columns[:-1], [0]*len(cpt.columns[:-1]))),False).replace(dict(zip(cpt.columns[:-1], [2]*len(cpt.columns[:-1]))),True).replace(dict(zip(cpt.columns[:-1], [1]*len(cpt.columns[:-1]))),True)#.drop(cpt.columns[-2],axis=1)

reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.mpe(evidence={'bowel-problem': True}))
# print(reasoner.map(query=['dog-out', 'hear-bark'],evidence={'family-out': True}))
# print(reasoner.marginalDistributions(query=['dog-out'],evidence={'dog-out': True}))
# print(reasoner.variableElimination(query=['dog-out'], evidence={'family-out': True}))

# reasoner = BNReasoner("./testing/work-from-home-problem.BIFXML")
# print(reasoner.mpe(query=['traffic'],evidence={'sick': True}))
# print(reasoner.map(query=['dog-out'],evidence={'dog-out': True}))
# print(reasoner.marginalDistributions(query=['dog-out'],evidence={'dog-out': True}))
# print(reasoner.variableElimination(query=['dog-out'], evidence={'family-out': True}))

'''
def mapping(self, query=dict(), evidence=dict()):
        order = self.Ordering('min-degree')

        order = [x for x in order if x not in query]
        print(order)

        for i in order:
            result = self.marginalization(i, self.bn.get_all_cpts()[i]) #can't check, marginalization doet het nog niet goed
            print('with marg:',result)
            self.bn.update_cpt(i, result)

        for q in query:
            domax = self.maxingOut(q, self.bn.get_all_cpts()[q])
            self.bn.update_cpt(q, domax)
            print(self.bn.get_cpt(q))

        print(domax.loc[:,'p'])
        print(domax.loc[:,'ins. of'])
'''