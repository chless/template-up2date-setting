import matplotlib
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import seaborn as sns


def image2neptune(some_plot):
    def inner(run: neptune.metadata_containers.run.Run, *args):
        fig = some_plot(run, *args)
        run.upload(neptune.types.File.as_image(fig))
        return some_plot(run, *args)

    return inner


def html2neptune(some_plot):
    def inner(run: neptune.metadata_containers.run.Run, *args):
        fig = some_plot(run, *args)
        run.upload(neptune.types.File.as_html(fig))
        return some_plot(run, *args)

    return inner


def plot_skeleton(run: neptune.metadata_containers.run.Run) -> matplotlib.figure.Figure:
    pass
    return


@image2neptune
def plot_grad(run, grads):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    grads = np.transpose(grads, axse=(1, 0))
    ax = sns.heatmap(grads, cmap="RdBu", cbar_kws={"label": "grad"})
    ax.set_xlabel("Steps")
    ax.set_ylabel("Layers")
    ax.set_title("Gradient flow")
    return fig
