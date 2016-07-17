import matplotlib.pyplot as plt
import seaborn as sns

from src.model.utils import *


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

    def update(self, nn, *args, **kwargs):
        sess = kwargs["sess"]
        data = kwargs["data"]
        step = kwargs["step"]

        summary, train_acc, train_loss = sess.run([nn.merged, nn.accuracy, nn.cost],
                                                  feed_dict=nn.feed_dict)
        # is it necessary to run rez after the above metrics?
        rez = sess.run(nn.output_softmax, feed_dict=nn.feed_dict)
        nn.train_writer.add_summary(summary, step)

        test_batch_size = 100
        batch_x, batch_y = data.train.next_batch(test_batch_size)
        feed_dict = {nn.x: batch_x, nn.y: batch_y, nn.keep_prob: 1}
        test_acc = sess.run(nn.accuracy, feed_dict=feed_dict)

        plot_data = [train_loss, train_acc, test_acc]
        self.observers.notify(nn, rez=rez, display_step=step, batch_size=test_batch_size, plot_data=plot_data)


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
            data.append(kwargs["plot_data"][i])
        print("Step " + str(step) + " Iter " + str(step * batch_size) +
              ", Train Loss= " + "{:.5f}".format(self.perf_data[0][-1]) +
              ", Train Accuracy= " + "{:.5f}".format(self.perf_data[1][-1]) +
              ", Test Accuracy= " + "{:.5f}".format(self.perf_data[2][-1]))


class NNOutputConsole:
    def __init__(self, koliko):
        self.rez_list = []
        self.koliko = koliko

    def update(self, nn, *args, **kwargs):
        self.rez_list.append(kwargs["rez"])
        print("Network output:\n", self.rez_list[-1][:self.koliko])
        print("---------\n")
