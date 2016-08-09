import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.model.utils import *
from src.data.data import *
from matplotlib import rcParams

sns.set_style("darkgrid", {'axes.grid': False})
rcParams.update({'figure.autolayout': True})


class Observer:
    def __init__(self, observers):
        self.observers = [i for i in observers]

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify(self, nn, *args, **kwargs):
        for observer in self.observers:
            observer.update(nn, *args, **kwargs)


class Plotter:
    def __init__(self, y=1, x=1):
        assert x >= 1 and y >= 1
        self.x = x
        self.y = y
        with sns.axes_style({"axes.grid": False}):
            self.fig, self.ax = plt.subplots(y, x)
        self.fig.show()
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')

    def update(self, images):
        if self.x == 1 and self.y == 1:
            self.ax.imshow(images[0][0])
        elif self.x == 1:
            for j in range(self.y):
                self.ax[j].imshow(images[j][0])
        elif self.y == 1:
            for i in range(self.x):
                self.ax[i].imshow(images[0][i])
        else:
            for i in range(self.x):
                for j in range(self.y):
                    self.ax[j][i].imshow(images[j][i])
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)


class ReconstructImage:
    def __init__(self, image, save_img=False, display_plot=True):
        self.image = np.expand_dims(image, axis=0)
        self.display_plot = display_plot
        self.save_img = save_img
        self.img_counter = 0
        if self.display_plot:
            self.plotter = Plotter(1, 2)

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        reconstructed_image = sess.run(nn.gen_output, feed_dict={nn.x: self.image})
        if self.display_plot:
            self.plotter.update([[normalize(self.image[0]), normalize(reconstructed_image[0])]])
        if self.save_img:
            save_folder = os.path.join(nn.tb_path, "reconstruction")
            image_name = nn.timestamp + "_" + str(self.img_counter)
            save_image(self.plotter.fig, save_folder, image_name)
            self.img_counter += 1


class TestEval:
    @staticmethod
    def update(nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        batch_x = data.test.next_batch(nn.hp.batch_size)
        feed_dict = {nn.x: batch_x}
        summary = sess.run(nn.merged, feed_dict=feed_dict)
        nn.test_writer.add_summary(summary, step)


class ReconstructRandomImages:
    def __init__(self, n_examples=5):
        self.n_examples = n_examples
        self.plotter = Plotter(2, self.n_examples)

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        nn.imgsum_gen = tf.image_summary("Generated_image" + str(step), nn.gen_output)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary = sess.run(nn.merged, feed_dict=nn.feed_dict, options=run_options, run_metadata=run_metadata)
        nn.train_writer.add_run_metadata(run_metadata, "step " + str(step))
        nn.train_writer.add_summary(summary, step)

        real = nn.feed_dict[nn.x]
        gen, doutput_gen, doutput_real = sess.run([nn.gen_output, nn.discr_on_gen, nn.discr_on_real],
                                                  feed_dict=nn.feed_dict)
        gen_img = normalize(gen)
        real_img = normalize(real)
        self.plotter.update([gen_img[:self.n_examples], real_img[:self.n_examples]])
        print("Output of the generator, generated:", np.mean(doutput_gen, axis=0), "    real:",
              np.mean(doutput_real, axis=0))


class ArithmPlotter:
    def __init__(self):
        starting_image_name = "075383.jpg"  # mladi muskarac sa sesirom
        image_to_subtract_name = "057434.jpg"  # mladi muskarac
        image_to_change_name = "057432.jpg"  # mlada zena koja se smije
        self.starting_image = np.expand_dims(CelebDataset.load_image(starting_image_name), axis=0)
        self.image_to_subtract = np.expand_dims(CelebDataset.load_image(image_to_subtract_name), axis=0)
        self.image_to_change = np.expand_dims(CelebDataset.load_image(image_to_change_name), axis=0)
        self.plotter = Plotter(1, 4)

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        starting_fs = sess.run(nn.z, feed_dict={nn.x: self.starting_image})[0, 0, 0, :]
        to_subtract_fs = sess.run(nn.z, feed_dict={nn.x: self.image_to_subtract})[0, 0, 0, :]
        to_change_fs = sess.run(nn.z, feed_dict={nn.x: self.image_to_change})[0, 0, 0, :]

        new_image = starting_fs - to_subtract_fs + to_change_fs
        img = np.reshape(new_image, [1, 1, 1, nn.autoenc_filter_depths[-1]])
        img = sess.run(nn.decoded_image, feed_dict={nn.z_input: img})[0, :, :, :]
        images = [self.starting_image[0], self.image_to_subtract[0], self.image_to_change[0], img]
        images = [normalize(img) for img in images]
        images = [images]
        self.plotter.update(images)


class MetricsConsole:
    def __init__(self):
        self.performance_data = []

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        summary, gen_loss, discr_loss, reconstr_loss = sess.run(
            [nn.merged, nn.gen_loss, nn.discr_loss, nn.reconstr_loss], feed_dict=nn.feed_dict)

        self.performance_data.append([reconstr_loss, gen_loss, discr_loss])
        print("-------------------------------------")
        print("Step " + str(step) + " Iter " + str(step * nn.hp.batch_size))
        print("Reconstr Loss= " + "{:.5f}".format(self.performance_data[-1][0]),
              ", Gen Loss= {:.5f}".format(self.performance_data[-1][1]),
              ", Discr Loss= {:.5f}".format(self.performance_data[-1][2]), end="\n -----------------------------\n ")


class PerformanceGraph:
    def __init__(self):
        n_metrics = 3
        self.perf_data = [[] for _ in range(n_metrics)]
        self.fig, self.ax1 = plt.subplots(figsize=(2, 1))
        self.ax2 = self.ax1.twinx()

        self.pallete = sns.color_palette("bright")
        self.font_size = 10
        # for item in ([self.ax1.title,
        #               self.ax1.xaxis.label, self.ax1.yaxis.label] +
        #                  self.ax1.get_xticklabels() + self.ax1.get_yticklabels()):
        for item in ([self.ax1.title,
                      self.ax1.xaxis.label, self.ax1.yaxis.label,
                      self.ax2.xaxis.label, self.ax2.yaxis.label] +
                         self.ax1.get_xticklabels() + self.ax1.get_yticklabels() +
                         self.ax2.get_xticklabels() + self.ax2.get_yticklabels()):
            item.set_fontsize(self.font_size)
        self.fig.show()

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]
        self.fig.canvas.draw_idle()
        plt.pause(0.00001)

        batch_size = nn.hp.batch_size
        test_batch = data.test.next_batch(nn.hp.batch_size)

        train_reconstr_loss = sess.run(nn.reconstr_loss, feed_dict=nn.feed_dict)
        test_reconstr_loss = sess.run(nn.reconstr_loss, feed_dict={nn.x: test_batch})

        append_data = [train_reconstr_loss, test_reconstr_loss, train_reconstr_loss]
        for i, data in enumerate(self.perf_data):
            data.append(append_data[i])
        self.ax1.set_xlabel("Training iterations: (x " + str(batch_size * step) + ")" +
                            "\nLearning rate: " + str(nn.hp.lr) +
                            "\nBatch size: " + str(batch_size))
        self.plot_data()
        if kwargs.get("save_fig", 0):
            save_folder = os.path.join(nn.tb_path, "visualization", "train_images")
            PerformanceGraph.save_image(save_folder, nn.timestamp)
            plt.close(self.fig)

    def plot_data(self):
        col1 = self.pallete[2]
        col2 = self.pallete[0]
        col3 = self.pallete[3]

        x_range = range(len(self.perf_data[0]))
        train_loss = self.ax1.semilogy(x_range, self.perf_data[0], color=col1, label="Train loss")
        test_loss = self.ax1.semilogy(x_range, self.perf_data[1], color=col2, label="Test loss")
        self.ax1.set_ylabel('Train/test loss', color=col1)
        for tl in self.ax1.get_yticklabels():
            tl.set_color(col1)

        # line2 = self.ax2.plot(x_range, self.perf_data[1], color=col2, label="Train accuracy")
        # self.ax2.set_ylabel("Train/validation accuracy", color=col2)
        # for tl in self.ax2.get_yticklabels():
        #     tl.set_color(col2)
        #
        # line3 = self.ax2.plot(x_range, self.perf_data[2], color=col3, label="Validation accuracy")

        lns = train_loss + test_loss
        labs = [l.get_label() for l in lns]
        self.ax1.legend(lns, labs, loc="lower right")
        plt.setp(self.ax1.get_title(), fontsize=self.font_size)

    @staticmethod
    def save_image(folder_path, image_name, dpi=300):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, image_name), dpi=dpi, bbox_inches="tight")
        print("Saved figure " + image_name + ".")
        plt.cla()
