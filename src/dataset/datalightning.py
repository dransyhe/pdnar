import os
import pickle as pkl
import lightning as L
import torch
import random
from tqdm import tqdm
from functools import partial
from torch_geometric.loader import DataLoader

from src.dataset.graph import generate_bipartite_graphs, generate_bipartite_graphs_from_barabasi, generate_test_graphs
from src.dataset.algorithms.hitting_set import hitting_set
from src.dataset.algorithms.vertex_cover import vertex_cover
from src.dataset.algorithms.set_cover import set_cover


class ParallelAlgData(L.LightningDataModule):
    def __init__(self,
                 algorithm, n_train_samples, n_val_samples, n_test_samples, n_train_nodes, ns_test_nodes,
                 batch_size, num_workers):
        super().__init__()
        self.algorithm = algorithm
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.n_train_nodes = n_train_nodes
        self.ns_test_nodes = ns_test_nodes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        if not os.path.exists(f"dataset/{self.algorithm}"):
            os.makedirs(f"dataset/{self.algorithm}")
        path = f"dataset/{self.algorithm}"

        if self.algorithm == 'hitting_set':
            generate_algorithmic_data = hitting_set 
            generate_graphs = generate_bipartite_graphs
            graph_type = 'hypergraph'
        elif self.algorithm == 'vertex_cover':
            generate_algorithmic_data = vertex_cover
            generate_graphs = generate_bipartite_graphs_from_barabasi
            graph_type = 'ordinary'
        elif self.algorithm == "set_cover":
            generate_algorithmic_data = set_cover
            generate_graphs = generate_bipartite_graphs
            graph_type = 'hypergraph' 
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm} is not implemented.") 

        # Training and validation sets
        if not (os.path.isfile(os.path.join(path, f"train_{self.n_train_samples}_{self.n_train_nodes}.pkl"))) and \
           not (os.path.isfile(os.path.join(path, f"val_{self.n_val_samples}_{self.n_train_nodes}.pkl"))):
            if not (os.path.isfile(os.path.join("dataset", f"train_{graph_type}.pkl"))):
                graphs = generate_graphs(self.n_train_samples + self.n_val_samples, self.n_train_nodes) 
                with open(os.path.join("dataset", f"train_{graph_type}.pkl"), 'wb') as fp:
                    pkl.dump(graphs, fp)
            else:
                with open(os.path.join("dataset", f"train_{graph_type}.pkl"), 'rb') as fp:
                    graphs = pkl.load(fp)
            data = [generate_algorithmic_data(graph) for graph in tqdm(graphs)]
            train_data = data[:self.n_train_samples]
            val_data = data[self.n_train_samples:]
            with open(os.path.join(path, f"train_{self.n_train_samples}_{self.n_train_nodes}.pkl"), 'wb') as fp:
                pkl.dump(train_data, fp)
            with open(os.path.join(path, f"val_{self.n_val_samples}_{self.n_train_nodes}.pkl"), 'wb') as fp:
                pkl.dump(val_data, fp)
        else:
            with open(os.path.join(path, f"train_{self.n_train_samples}_{self.n_train_nodes}.pkl"), 'rb') as fp:
                train_data = pkl.load(fp)
            with open(os.path.join(path, f"val_{self.n_val_samples}_{self.n_train_nodes}.pkl"), 'rb') as fp:
                val_data = pkl.load(fp)
        self.train_data = train_data
        self.val_data = val_data

        # Testing datasets of different graph sizes
        all_test_data = []
        for n_test_nodes in self.ns_test_nodes:
            if not os.path.isfile(os.path.join(path, f"test_{self.n_test_samples}_{n_test_nodes}.pkl")):
                if not (os.path.isfile(os.path.join("dataset", f"test_{graph_type}_{n_test_nodes}.pkl"))):
                    graphs = generate_graphs(self.n_test_samples, n_test_nodes) 
                    with open(os.path.join("dataset", f"test_{graph_type}_{n_test_nodes}.pkl"), 'wb') as fp:
                        pkl.dump(graphs, fp)
                else:
                    with open(os.path.join("dataset", f"test_{graph_type}_{n_test_nodes}.pkl"), 'rb') as fp:
                        graphs = pkl.load(fp)
                test_data = [generate_algorithmic_data(graph) for graph in tqdm(graphs)]
                with open(os.path.join(path, f"test_{self.n_test_samples}_{n_test_nodes}.pkl"), 'wb') as fp:
                    pkl.dump(test_data, fp)
            else:
                with open(os.path.join(path, f"test_{self.n_test_samples}_{n_test_nodes}.pkl"), 'rb') as fp:
                    test_data = pkl.load(fp)
            all_test_data.append(test_data)

        self.test_dataset = all_test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, follow_batch=['x', 'y'])

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, follow_batch=['x', 'y'])

    def test_dataloader(self):
        dataloaders = []
        for test_data in self.test_dataset:
            dataloader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_workers, follow_batch=['x', 'y'])
            dataloaders += [dataloader]
        return dataloaders


if __name__ == '__main__':
    datalight = ParallelAlgData(
        n_train_samples=100,
        n_val_samples=5,
        n_test_samples=5,
        n_train_nodes=16,
        ns_test_nodes=[16, 32, 64],
        batch_size=32,
        algorithm='hitting_set',
        num_workers=0)
    datalight.prepare_data()



