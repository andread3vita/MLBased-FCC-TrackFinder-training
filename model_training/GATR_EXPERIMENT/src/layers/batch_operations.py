import dgl
import torch


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    list_y = add_batch_number(list_graphs)
    ys = torch.cat(list_y, dim=0)
    ys = torch.reshape(ys, [-1, list_y[0].shape[1]])

    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys


def add_batch_number(list_graphs):
    list_y = []
    for i, el in enumerate(list_graphs):
        y = el[1]
        batch_id = torch.ones(y.shape[0], 1) * i
        y = torch.cat((y, batch_id), dim=1)
        list_y.append(y)
    return list_y


def obtain_batch_numbers(g):
    dev = g.ndata["pos_hits_xyz"].device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch
