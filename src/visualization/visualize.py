import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.model.utils import *


sns.set_style("whitegrid", {'axes.grid' : False})
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


class AllMetrics:
    def __init__(self, observers=None):
        self.observers = observers
        self.n_examples = 5
        self.fig, self.ax = plt.subplots(2, self.n_examples)
        self.fig.show()

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        # summary, train_acc, gen_loss = sess.run([nn.merged, nn.accuracy, nn.cost],
        #                                           feed_dict=nn.feed_dict)
        nn.imgsum_gen = tf.image_summary("Generated_image" + str(step), nn.gen_output)
        summary, gen_loss, discr_loss, _, _ = sess.run(
            [nn.merged, nn.gen_loss, nn.discr_loss, nn.imgsum_real, nn.imgsum_gen],
            feed_dict=nn.feed_dict)
        train_acc = -1
        nn.train_writer.add_summary(summary, step)
        rez, test_batch_size = -1, nn.hyp_param["batch_size"]

        ind = 1
        gen, real, doutput_gen, doutput_real = sess.run([nn.gen_output, nn.x, nn.discr_on_gen, nn.discr_on_real],
                                                        feed_dict=nn.feed_dict)
        gen_img = gen / 2 + 0.5
        real_img = real / 2 + 0.5
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)
        # plt.draw()
        # plt.show()
        for i in range(self.n_examples):
            self.ax[0][i].imshow(gen_img[ind + i])
            self.ax[1][i].imshow(real_img[ind + i])
        # plt.waitforbuttonpress()
        print("Output of the generator, generated:", doutput_gen[ind], "    real:", doutput_real[ind])

        plot_data = [gen_loss, discr_loss, discr_loss < gen_loss]
        self.observers.notify(nn, rez=rez, display_step=step, batch_size=test_batch_size, plot_data=plot_data)

    def plotimg(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]
        ind = kwargs["ind"]
        slika = sess.run(nn.gen_output, feed_dict=nn.feed_dict)
        img = slika[ind, :, :, :]
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.show()
        plt.draw()
        plt.show()
        plt.waitforbuttonpress()


class PerformanceGraph:
    def __init__(self):
        n_metrics = 3
        self.perf_data = [[] for _ in range(n_metrics)]
        self.fig, self.ax1 = plt.subplots(figsize=(2, 1))
        self.ax2 = self.ax1.twinx()

        sns.set_style("darkgrid")
        self.pallete = sns.color_palette("bright")
        self.font_size = 10
        for item in ([self.ax1.title,
                      self.ax1.xaxis.label, self.ax1.yaxis.label,
                      self.ax2.xaxis.label, self.ax2.yaxis.label] +
                         self.ax1.get_xticklabels() + self.ax1.get_yticklabels() +
                         self.ax2.get_xticklabels() + self.ax2.get_yticklabels()):
            item.set_fontsize(self.font_size)
        self.fig.show()

    def update(self, nn, *args, **kwargs):
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)
        batch_size = kwargs["batch_size"]
        display_step = kwargs["display_step"]
        for i, data in enumerate(self.perf_data):
            data.append(kwargs["plot_data"][i])
        self.ax1.set_xlabel("Training iterations: (x " + str(batch_size * display_step) + ")" +
                            "\nLearning rate, decay, decay_steps_div: " + str(nn.hyp_param["learning_rate"]) + ", " +
                            str(0) + ", " + str(0) +
                            "\nBatch size: " + str(batch_size))
        self.plot_data()
        if kwargs.get("save_fig", 0):
            save_folder = os.path.join(nn.tb_path, "visualization", "train_images")
            PerformanceGraph.save_image(save_folder, nn.timestamp)
            plt.close(self.fig)

    def plot_data(self):
        col = self.pallete[2]

        x_range = range(len(self.perf_data[0]))
        line1 = self.ax1.semilogy(x_range, self.perf_data[0], color=col, label="Train loss")
        self.ax1.set_ylabel('Train loss', color=col)
        for tl in self.ax1.get_yticklabels():
            tl.set_color(col)

        col = self.pallete[0]
        line2 = self.ax2.plot(x_range, self.perf_data[1], color=col, label="Train accuracy")
        self.ax2.set_ylabel("Train/validation accuracy", color=col)
        for tl in self.ax2.get_yticklabels():
            tl.set_color(col)

        line3 = self.ax2.plot(x_range, self.perf_data[2], color=self.pallete[3], label="Validation accuracy")

        lns = line1 + line2 + line3
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


class MetricsConsole:
    def __init__(self):
        n_metrics = 3
        self.perf_data = [[] for _ in range(n_metrics)]

    def update(self, nn, *args, **kwargs):
        step = kwargs["display_step"]
        batch_size = kwargs["batch_size"]
        for i, data in enumerate(self.perf_data):
            data.append(kwargs.get("plot_data", -1)[i])
        print("Step " + str(step) + " Iter " + str(step * batch_size) +
              ", Gen Loss= " + "{:.5f}".format(self.perf_data[0][-1]),
              ", Discr Loss= {:.5f}".format(self.perf_data[1][-1]), end=" ")
        if self.perf_data[0][-1] > self.perf_data[1][-1]:
            print("Generator")
        else:
            print("Discriminator")

    class NNOutputConsole:
        def __init__(self, koliko):
            self.rez_list = []
            self.koliko = koliko

        def update(self, nn, *args, **kwargs):
            self.rez_list.append(kwargs["rez"])
            print("Network output:\n", self.rez_list[-1][:self.koliko])
            print("---------\n")
