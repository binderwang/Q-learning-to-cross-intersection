import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


fc='black'
cm='gray'
fig, ax = plt.subplots(1, 1)#, figsize=(7,7))
fig.patch.set_facecolor(fc)

vmin = 0
vmax = 3
def show_img_return_input(img, name, cm='gray', ask=True):
    plt.ion()
    plt.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
    plt.show()
    ax.spines['bottom'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    ax.xaxis.label.set_color('red')
    plt.xticks([])
    plt.yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if type(name)==int:
        ax.set_title('frame '+str(name))
    else:
        ax.set_title(name)
    ax.title.set_color('red')
    plt.tight_layout()
    plt.draw()
    if ask:
        accept = input('OK? ')
    else:
        accept = 'y'
        plt.pause(0.1)
    plt.cla()
    return(accept)


def write_img(img, name, out_dir, cm='gray'):
    plt.ion()
    plt.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
    ax.spines['bottom'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    ax.xaxis.label.set_color('red')
    plt.xticks([])
    plt.yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if type(name)==int:
        ax.set_title('frame '+str(name))
    else:
        ax.set_title(name)
    ax.title.set_color('red')
    plt.savefig(out_dir+'/'+name, bbox_inches='tight')