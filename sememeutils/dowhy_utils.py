import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm

# Utility functions for DoWhy-based pruning
gcm.util.general.set_random_seed(0)

def node_name(block_idx):
    return f'B{block_idx}'

def create_nx_causal_graph(losses_dict):
    graph_items = []
    for node in [node_name(i) for i in losses_dict.keys()]:
        graph_items.append((node, 'loss'))
    causal_graph = nx.DiGraph(graph_items)
    return causal_graph

def convert_losses_to_df(losses_dict):
    all_nodes = [node_name(i) for i in losses_dict.keys()]
    data = []
    for block_idx, losses in losses_dict.items():
        one_cold = {node: 1 for node in all_nodes}
        one_cold[node_name(block_idx)] = 0  # Set the removed block to 0
        for loss in losses:
            data_d = one_cold.copy()
            data_d['loss'] = loss
            data.append(data_d)
    df = pd.DataFrame(data)
    return df

def arg_bot_k(arr, k):
    return np.argsort(np.array(arr))[:k]

def get_avg_sample_loss(causal_model, sample_losses, num_samples_to_draw=1000):
    avg_losses = {}
    for i in sample_losses.keys():
        node = node_name(i)
        samples = gcm.interventional_samples(causal_model, {node: lambda y: 0}, num_samples_to_draw=num_samples_to_draw)
        mean_loss = samples.loss.mean()
        avg_losses[i] = mean_loss
    return avg_losses

def fit_causal_model(samples_losses):
    sampled_data = convert_losses_to_df(samples_losses)
    causal_graph = create_nx_causal_graph(samples_losses)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    gcm.auto.assign_causal_mechanisms(causal_model, sampled_data)
    gcm.fit(causal_model, sampled_data)
    return causal_model

def get_next_block_to_prune(samples_losses, num_samples_to_draw=1000, return_predicted_losses=False):
    causal_model = fit_causal_model(samples_losses)
    predicted_losses = get_avg_sample_loss(causal_model, samples_losses, num_samples_to_draw=num_samples_to_draw)
    idx_mapping = []
    predicted_losses_array = []
    for block_idx, pred_loss in predicted_losses.items():
        predicted_losses_array.append(pred_loss)
        idx_mapping.append(block_idx)
    worst_idx = arg_bot_k(np.array(predicted_losses_array), 1)[0].item()
    if return_predicted_losses:
        return idx_mapping[worst_idx], predicted_losses
    return idx_mapping[worst_idx]
