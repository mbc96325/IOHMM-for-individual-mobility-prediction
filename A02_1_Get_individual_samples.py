import pickle
import numpy as np
with open ('../data/individual_ID_list', 'rb') as fp:
    individual_ID_list = pickle.load(fp)

with open('../data/not_recommend_using_individual_ID.pickle', 'rb') as fp: # due to high error records rate
    not_recommend_samples = pickle.load(fp)

seed = 11
Num_ind = 1000

print('Total ind before', len(individual_ID_list))
individual_ID_list = list(set(individual_ID_list).difference(not_recommend_samples))
print('After filter not recommend ind', len(individual_ID_list))

np.random.seed(seed)
used_individual = list(np.random.choice(individual_ID_list, size=Num_ind, replace=False))

assert len(set(used_individual)) == len(used_individual) # no replacement check

analysis_individual = [994326032]

for idx in analysis_individual:
    if idx not in used_individual:
        used_individual.append(idx)
        used_individual.pop(0)
assert len(used_individual) == Num_ind

print('Final num selected sample', len(used_individual))

with open ('../data/individual_ID_list_test_' + str(Num_ind) + '.pickle', 'wb') as fp:
    pickle.dump(used_individual,fp)

