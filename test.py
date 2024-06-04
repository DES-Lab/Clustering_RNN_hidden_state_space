import pickle

with open('experiment_results/new_ambiguity_results_top_of_stack.pickle', 'rb') as handle:
    data = pickle.load(handle)

new_data = dict()
for k, v in data.items():
    processed_data = dict()
    for stat in v:
        key = stat[0]
        processed_data[key] = [stat[1][0], stat[1][1], stat[1][2], stat[1][3], stat[2]]
    new_data[k] = processed_data

print(new_data)
#
with open('experiment_results/new_ambiguity_results_top_of_stack.pickle', 'wb') as handle:
    pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)