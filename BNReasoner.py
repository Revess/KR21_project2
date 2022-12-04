from typing import Union
from BayesNet import BayesNet
import pandas as pd

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
        print(evidence)
        cpts = self.bn.get_all_cpts()
        initiations = pd.Series(evidence)
        # self.bn.draw_structure()

        # First reduce factors
        for node in cpts.keys():
            if sum([1 if ev in cpts[node].keys() else 0 for ev in evidence]) >= 1:
                newCPT = self.bn.reduce_factor(initiations,cpts[node])
                newCPT = newCPT[newCPT.p != 0]
                self.bn.update_cpt(node, newCPT)

        # Then prune the edges and nodes
        for ev in evidence.keys():
            for child in self.bn.get_children(ev):
                self.bn.del_edge((ev,child))
                newCPT = self.bn.get_cpt(child).drop(ev,axis=1)
                self.bn.update_cpt(child, newCPT)
                if newCPT.drop('p',axis=1).columns.shape[0] <= 1:
                    self.bn.del_var(child)
            self.bn.del_var(ev)
        self.bn.draw_structure()
    def maxingOut(self, variable):
        cpts = self.bn.get_all_cpts()
        df = cpts[variable]
        res = pd.DataFrame(columns=df.columns.drop([variable]))

        for i in range(len(df['family-out'])):
            if i % 2 == 0:
                max = df.loc[i:i, ['p', variable]]
            else:
                if df.loc[i, 'p'] > max.iloc[0, 0]:
                    max = df.loc[i:i, ['p', variable]]
                maxres = df.drop([variable, 'p'], axis=1).loc[i:i, :]
                maxres['p'] = max.iloc[0, 0]
                maxres['ins. of ' + variable] = max.iloc[0, 1]
                res = pd.concat([res, maxres], axis=0, sort=False, ignore_index=True)
        return res

    def factorMultiplication(self, factor1, factor2):
        cpts = reasoner.bn.get_all_cpts()
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

reasoner = BNReasoner("./testing/dog_problem.BIFXML")
a = reasoner.bn.get_interaction_graph()

