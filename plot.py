import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, type='original', percentage=5):
        self.type = type
        self.percentage = percentage

    def plot_bar(self, stats_list, total_mean_list, model_name_list, metric_name, type):
        plt.rcParams["figure.figsize"] = (14,9)
        plt.rcParams["font.size"] = 17
        colors = ['red', 'green', 'grey', 'orange']
        line_styles = ['--', '--', '--', '--']
        distances = [-0.3, -0.1, 0.1, 0.3]

        ax = plt.subplot(111)

        for counter, model in enumerate(zip(stats_list, total_mean_list, model_name_list),0):
            stats = model[0]
            total_mean = model[1]
            model_name = model[2]
            print(counter,model_name)

            objects = list(stats.keys())
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

        if self.type == 'original':
            plt.savefig('plots/{}.jpg'.format(metric_name))
        else:
            plt.savefig('plots/splitted/{}_{}_{}.jpg'.format(metric_name,str(self.percentage),type))

        plt.clf()
