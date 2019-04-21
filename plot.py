import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        pass

    def plot_bar(self, stats_list, total_mean_list, model_name_list, metric_name):
        plt.rcParams["figure.figsize"] = (16,10)
        colors = ['r','g', 'grey']
        line_styles = ['--', '--', '--']
        distances = [-0.3, 0, 0.3]

        ax = plt.subplot(111)

        for counter, model in enumerate(zip(stats_list, total_mean_list, model_name_list),0):
            stats = model[0]
            total_mean = model[1]
            model_name = model[2]

            objects = list(stats.keys())
            # print(objects)
            y_pos = np.arange(len(objects))
            mean = []
            for list_of_values in stats.values():
                mean.append(np.mean(list_of_values))


            ax.bar(y_pos+distances[counter], mean, width=0.3, color=colors[counter], align='center', label='{}'.format(model_name))
            plt.axhline(y=float(total_mean/100), color=colors[counter], linestyle=line_styles[counter])

        plt.xticks(y_pos, objects, rotation=45)
        plt.ylabel(metric_name)
        plt.xlabel('Question type')
        plt.title(metric_name)
        plt.legend()

        plt.savefig('{}.jpg'.format(metric_name))

        plt.clf()