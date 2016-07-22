from src.model.convolutional_network import *

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hyperparameters = {"learning_rate": 0.0002,
                   "beta1": 0.5,
                   "training_iters": 10000000,
                   "batch_size": 100,
                   "n_input": 784,
                   "n_classes": 10}

metrics = AllMetrics(Observer([MetricsConsole()]))
observers = Observer([metrics])

old_model_path_petnica = "/home/bgavran3/petnica/tf_logs/2016_July_22__03:52.tmp"
nn = DCGAAE(hyperparameters, observers=observers)
nn.train(CelebDataset(), hyperparameters["batch_size"], hyperparameters["training_iters"],
         old_model_path=old_model_path_petnica)
# test_accuracy = nn.test(mnist, 256, nn.model_path)

# print("Test accuracy:", test_accuracy)
# nn.plot_filter(nn._weights["wc1"])
# nn.plot_filter(nn._weights["wc2"])
# nn.plot_filter(nn._weights["wc3"])
