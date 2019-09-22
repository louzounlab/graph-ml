# Topology-Based Graph Convolutional Networks

Node classification in graphs using topology-based graph convolutional network (GCN).

Code file for the paper:
"Topological based classification of paper domains using graph convolutional networks" by Roy Abel, Idan Benami and Yoram Louzoun.

To quickly run an experiment set the parameters at the top of the code:
* DataSetName to be one of the above: "Cora", "CiteSeer", "PubMed", "CoraFull", "CS", "Physics"
* Net to be one of the following models: "gcn", "siam", "gat", "siam_gat", "combined", "symmetric", "asymmetric"
* Split to determine how to split the train set: "standard", "20_30" , "percent", None
and run main.py.

## Creating new experiment

create parameters dictionary:
params_ = {"data_name": DataSetName,

                  "net": ,            		# available networks are: "gcn", "siam", "gat", "siam_gat", "combined", "symmetric", "asymmetric"
                  "is_nni": False,          # should be True only if running through Microsoft NNI platform
                  "verbose": 1,             # verbose 2 will print scores after each epoch, verbose 1 will print scores only after all epochs
                  "trials": ,         		# number of trials to run. output is the average score over all trials
                  "neighbors_ft": ,   		# set True to use neighbors's labels feature instead of standard topological features
                  "knn": ,          		# number of neighbors in the topological graph
                  "directed": False,		# whether the graph is the graph directed or not

                  "epochs": ,
                  "activation": ,			# "relu", "elu", "tanh"
                  "dropout_rate": ,
                  "hidden_sizes": ,
                  "learning_rate": ,
                  "weight_decay": ,

                  "norm": ,            		# normalize input features. for gat architectures should be True
                  "gat_heads": ,     		# number of heads at each layers - only for GAT architectures
                }
More details can bound at the paper: "Topological based classification of paper domains using graph convolutional networks"

Create a Model object with the desired parameters

Load data set using Model.load_data()

Build a GCN network using Model.build_architecture()

Train Model's parameters using Model.train()

Test the model with Model.test()


## Code Example

```python

if __name__ == '__main__':
	params_ = update_params(params_, net=Net, data_name=DataSetName)
	model = Model(params_)
    model.load_data()
	
	acc = []
	for _ in range(parameters['trials']):
        model.build_architecture()
        model.train()
        acc.append(model.test())
	
	avg_acc = np.mean(acc)
	Logger.info("average test acc: {:.3f}% \n std is: {}".format(avg_acc * 100, np.std(acc) * 100))
	
```

