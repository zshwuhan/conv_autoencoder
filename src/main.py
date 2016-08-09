from src.model.convolutional_network import *
from src.visualization.visualize import *


class Hyperparameters:
    lr = 0.0003
    training_iters = 10000000
    batch_size = 100
    input_shape = (64, 64, 3)
    autoenc_filter_depths = (3, 128, 128, 256, 256, 512, 1024)
    autoenc_filter_sizes = (-1, 5, 3, 3, 3, 3, 2)
    discr_filter_depths = (64, 128, 256, 1024)
    discr_filter_sizes = (5, 3, 3, 3)


img_path = "/home/bgavran3/petnica/src/model/downloaded_examples/DCGAN_tensorflow/data/bare_slozen.jpg"
# recon = ReconstructImage(CelebDataset.load_image(img_path, is_crop=True), save_img=False)
observers = Observer(
    [ReconstructRandomImages(), TestEval, MetricsConsole()])
nn = DCGAAE(Hyperparameters, observers=observers, corrupt=ImageCorruption.noise)
data = DataSets(CelebDataset)

# old_model_path = "/home/bgavran3/petnica/tf_logs/2016_August_07__20:10.tmp"
nn.train(data, old_model_path=None, display_step=100)
