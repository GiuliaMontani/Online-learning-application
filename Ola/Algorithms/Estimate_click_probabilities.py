import numpy as np

#estimate probability to visit a node starting from another one

def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob = np.ones(n_nodes)*1.0/(n_nodes-1)
    credits = np.zeros(n_nodes)
    occur_v_active = np.zeros(n_nodes)
    #n_episodes = len(dataset)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] ==1).reshape(-1)
        if len(idx_w_active)>0 and np.any(idx_w_active>0):
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
            credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
        for v in range(0,n_nodes):
            if(v!=node_index):
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if (len(idx_v_active)>0 and len(idx_w_active)==0) or (np.any(idx_v_active<idx_w_active) and len(idx_v_active)>0):
                    occur_v_active[v]+=1
    estimated_prob = np.nan_to_num(credits/occur_v_active)
    return estimated_prob