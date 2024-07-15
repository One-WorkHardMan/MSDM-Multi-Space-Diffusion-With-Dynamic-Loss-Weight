from guided_diffusion.image_datasets import load_data
from model_prep import pre_trained_models
from guided_diffusion.unet import perpectual_Encoder_model
import blobfile as bf
import torch
import numpy as np
from sklearn.decomposition import PCA

data_dir ="/data/students/liuzhou/projects/DataSets/tinyimagenet/tiny-imagenet-200/train"
batch_size = 1
image_size = 64
dir = "/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_stage-one-812-1-tinyimagenet-vit_b_layer--lr-1e-4-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_2"

data = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=None,
    # deterministic=True #测试用，马上删掉
)
print(torch.cuda.is_available())

batch, cond = next(data)
model = perpectual_Encoder_model(image_size=image_size, out_space_channels=512)

with bf.BlobFile(f"{dir}/_modelname__Encoder_050000.pt", "rb") \
        as f:
    with torch.no_grad():
        state_dict = torch.load(f, map_location="cpu")
        model.load_state_dict(state_dict)

output = model(batch)
print(output.shape)
spaces = []
for i,start in zip(range(3),np.linspace(0,3,5)):
    spaces.append(output.chunk(3, dim=1)[i])
    spaces[i] = spaces[i].reshape(1,512,-1)
    spaces[i] = torch.squeeze(spaces[i],0)
    print(spaces[i].shape)
    # np.savetxt(f"space{i}.csv",spaces[i].detach().numpy(),delimiter=',')
    # torch.save(spaces[i],f"space{i}.csv")
    pca = PCA(n_components=2)
    spaces[i] = spaces[i].reshape(512, -1)
    # print(spaces[0].shape)
    x_reduced = pca.fit_transform(spaces[i].detach().numpy())
    print(x_reduced.shape)
    np.savetxt(f"pac_space{i}.csv", x_reduced, delimiter=',')

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)

    # Rotate the starting point around the cubehelix hue circle
    # for ax, s in zip(axes.flat, 0.3):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=start, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = x_reduced[:, 0], x_reduced[:, 1]
    sns.kdeplot(
        x=x, y=y,
        cmap=cmap, fill=True,
        clip=(-5, 5), cut=10,
        thresh=0, levels=15,
        ax=axes,
    )
    # axes.set_axis_off()
    axes.set(xlim=(-4, 4), ylim=(-4, 4))
    # f.subplots_adjust(0, 0, 1, 1, .08, .08)
    plt.savefig(f"myfig_kdf{i}.png")














