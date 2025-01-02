from mnist import CNN, test
import torch
from torch import nn
from mnist_data import make_test_data_loader
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def main():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    loader = make_test_data_loader()
    index = 1024
    #print(loader.dataset[index][1])
    #plt.imshow(loader.dataset[index][0][0], cmap="gray")
    #plt.show()
    #test(loader, model, loss_fn, device)

    model.eval()
    with torch.no_grad():
        #image = loader.dataset[index][0].to(device)
        #activations = model.activations(image)
        #activation = activations[4]
        #print(activation.shape)
        #print(activation)
        matrices = np.squeeze(next(model.conv2.parameters()).data)
        print(matrices.shape)
        matrices = matrices[0]
        # Initial slice to display
        slice_index = 0

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(matrices[slice_index])#, cmap='gray')
        plt.title(f"Slice {slice_index}")
        plt.axis('off')

        # Add a slider below the plot
        ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])  # x, y, width, height
        slider = Slider(ax_slider, "Slice", 0, matrices.shape[0] - 1, valinit=slice_index, valstep=1)

        # Update function for slider
        def update(val):
            slice_idx = int(slider.val)
            im.set_data(matrices[slice_idx])
            ax.set_title(f"Slice {slice_idx}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()


if __name__ == "__main__":
    main()
