from BNReasoner import BNReasoner

q=["dog-out"]
e={"family-out":True}
print("Pruning Network")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
reasoner.pruneNetwork(Q=q, evidence=e)

print("Reducing Network")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
reasoner.reduceNet(evidence=e)

print("Maxing Out")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.maxingOut(variable="family-out", cpt=reasoner.bn.get_all_cpts()["family-out"]))

print("Factor Multiplication")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.factorMultiplication(f1=reasoner.bn.get_all_cpts()["family-out"], f2=reasoner.bn.get_all_cpts()["bowel-problem"]))

print("Ordering")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.Ordering(heuristic='min-fill'))

print("Variable Elimination")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.variableElimination(query=q, evidence=e))

print("Marginal Distribution")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.marginalDistributions(query=q, evidence=e))

print("MAP")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.map(query=q, evidence=e))

print("MPE")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.mpe(evidence=e))

print("D-Seperation")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.dSeperation(X=["family-out"], Y=["bowel-problem"], Z=["dog-out"]))

print("Indepenedence")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.independence(X=["family-out"], Y=["bowel-problem"], Z=["dog-out"]))

print("Marginalization")
reasoner = BNReasoner("./testing/dog_problem.BIFXML")
print(reasoner.marginalization(cpt=reasoner.bn.get_all_cpts()["dog-out"], sumFor="family-out"))