import matplotlib.pyplot as plt 
import digits_data as dd 
from visualize_maps import kernel_one, fm_0

fig, axs = plt.subplots(6, 3)

im00 = axs[0, 0].imshow(dd.image_train[0], cmap='binary')
fig.colorbar(im00, ax=axs[0,0])
im01 = axs[0, 1].imshow(kernel_one, cmap='binary') #don't want to norm this one bc the neg values are valueable
fig.colorbar(im01, ax=axs[0,1])
im02 = axs[0, 2].imshow(fm_0, cmap='binary', vmin=0, vmax=1)
fig.colorbar(im02, ax=axs[0,2])
plt.show()
