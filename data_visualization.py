import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

from data_loading import load_data

def show(x, outfile = False, figsize_=(20,20), num_img=25, height=5, width=5):
    """
    Data visualization of test data

    Args:
        x (numpy array): Test dataset
        outfile (boolean, optional): Specify wether storing generated visualization is needed. Defaults to False.
        figsize_ (tuple, optional): Size of generated figure. Defaults to (20,20).
        num_img (int, optional): Number of images containing in the figure. Defaults to 25.
        height (int, optional): Number of rows. Defaults to 5.
        width (int, optional): Number of colums. Defaults to 5.
    """    

    plt.figure(figsize=figsize_)
    for i in range(num_img):
        plt.subplot(height,width,i+1)  
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
    
    if outfile == True:
        # Check if the figure folder exists
        if not os.path.exists(os.getcwd()+"/figure"):
            os.makedirs(os.getcwd()+"/figure")
        plt.savefig("./figure/data_visualization.png")
        
    plt.show()


def show_x_t(X_t, outfile=None, figsize_=(100, 100), num_img=25, height=5, width=5):
    
    fig = plt.figure(figsize=figsize_)
    grid = ImageGrid(
        fig, 111, 
        nrows_ncols=(len(X_t[0]), 2),
        axes_pad=0.1
    )

    for i in range(0, len(grid), 2):
        grid[i].imshow(X_t[0][int(i/2)].squeeze())
        grid[i+1].imshow(X_t[1][int(i/2)].squeeze())

    if outfile:
        # Check if the figure folder exists
        if not os.path.exists(os.getcwd()+"/figure"):
            os.makedirs(os.getcwd()+"/figure")
        plt.savefig(f"./figure/{outfile}")
        
    plt.show()

def plot_pair(image, mask):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 1
    for i in range(1, columns*rows +1):
        img = image if i == 1 else mask
        img = img.cpu().swapaxes(0,-1).swapaxes(0,1)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    data_loader, _ = load_data('isbi_em_seg', n_train=30, n_test=0, batch_size=30)
    data_set = next(iter(data_loader)) 

    show_x_t(data_set, 'image_double.png')