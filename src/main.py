from src.model.convolutional_network import *

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hyperparameters = {"learning_rate": 0.001,
                   "training_iters": 2000000,
                   "batch_size": 100,
                   "n_input": 784,
                   "n_classes": 10,
                   "dropout": 0.75}

metrics = AllMetrics(Observer([NNOutputConsole(1), CostConsole()]))

nn = CNN(hyperparameters, Observer([metrics]))
nn.train(mnist, hyperparameters["batch_size"], hyperparameters["training_iters"])
test_accuracy = nn.test(mnist, 256, nn.model_path)

print("Test accuracy:", test_accuracy)
# nn.plot_filter(nn.conv2)
