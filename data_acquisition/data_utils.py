from cProfile import label
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from pyparsing import col
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def reproject_img(src, dst_path, dst_crs='EPSG:3006'):
    '''
    Reprojects the rasterio dataset src and saves it to <dataset_path>\\reprojected
    '''
    
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()

    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)


def plot_masked_image(img, mask, mask_code=None):
    if mask_code is not None:
        mask = np.array([[1 if i == mask_code else 0 for i in j] for j in mask])

    def plot_masked_rgb(red, green, blue, mask, color_mask=(1, 0, 0), transparency=0.5, brightness=1):
        
        # to improve our visualization, we will increase the brightness of our values
        max_value = np.max([red.max(), green.max(), blue.max()])
        red = red / max_value * brightness
        green = green / max_value * brightness
        blue = blue / max_value * brightness
        
        red = np.where(mask==1, red*transparency+color_mask[0]*(1-transparency), red)
        green = np.where(mask==1, green*transparency+color_mask[1]*(1-transparency), green)
        blue = np.where(mask==1, blue*transparency+color_mask[2]*(1-transparency), blue)
        
        rgb = np.stack([red, green, blue], axis=2)
        
        return rgb

    masked_rgb = plot_masked_rgb(red=img[:, :, 3],
                                green=img[:, :, 2],
                                blue=img[:, :, 1],
                                mask=mask,
                                color_mask=(0, 0, 1),
                                brightness=1
                                )

    plt.figure(figsize=(15, 15))
    plt.imshow(masked_rgb)
    plt.show()


land_cover_map_list = [
    [111, 'Pine forest not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10% where >70% of the crown cover consists of pine. Trees are higher than 5 meters', (110, 140, 5)],
    [112, 'Spruce forest not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10% where >70% of the crown cover consists of spruce. Trees are higher than 5 meters', (45, 95, 0)],
    [113, 'Mixed coniferous not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10% where >70% of consists of pine or spruce, but none of these species are >70%. Trees are higher than 5 meters.', (78, 112, 0)],
    [114, 'Mixed forest not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10% where neither coniferous nor deciduous crown cover reaches >70%. Trees are higher than 5 meters.', (56, 168, 0)],
    [115, 'Deciduous forest not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10% where >70% of the crown cover consists of deciduous trees (primarily birch, alder and/or aspen). Trees are higher than 5 meters.', (76, 230, 0)],
    [116, 'Deciduous hardwood forest not on wetland', 'Tree-covered areas outside of wetlands with a total crown cover of >10 where >70% of the crown cover consists of deciduous trees, of which >50% is broad-leaved deciduous forest (mainly oak, beech, ash, elm, linden, maple, cherry and hornbeam).Trees are higher than 5 meters.', (170, 255, 0)],
    [118, 'Temporarily non-forest not on wetland', 'Open and re-growing clear-felled, storm-felled or burnt areas outside of wetlands. Trees are less than 5 meters.', (205, 205, 102)],
    [121, 'Pine forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10% where >70% of the crown cover consists of pine. Trees are higher than 5 meters', (89, 140, 85)],
    [122, 'Spruce forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10% where >70% of the crown cover consists of spruce. Trees are higher than 5 meters', (48, 94, 80)],
    [123, 'Mixed coniferous on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10% where >70% of consists of pine or spruce, but none of these species are >70%. Trees are higher than 5 meters.', (35, 115, 90)],
    [124, 'Mixed forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10% where neither coniferous nor deciduous crown cover reaches >70%. Trees are higher than 5 meters.', (67, 136, 112)],
    [125, 'Deciduous forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10% where >70% of the crown cover consists of deciduous trees (primarily birch, alder and/or aspen). Trees are higher than 5 meters.', (137, 205, 155)],
    [126, 'Deciduous hardwood forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10 where >70% of the crown cover consists of deciduous trees, of which >50% is broadleaved deciduous forest (mainly oak, beech, ash, elm, linden, maple, cherry and hornbeam). Trees are higher than 5 meters.', (165, 245, 120)],
    [127, 'Deciduous forest with deciduous hardwood forest on wetland', 'Tree-covered areas on wetlands with a total crown cover of >10 where >70% of the crown cover consists of deciduous trees, of which 20 - 50% is broadleaved deciduous forest (mainly oak, beech, ash, elm, linden, maple, cherry and hornbeam). Trees are higher than 5 meters.', (171, 205, 120)],
    [128, 'Temporarily non-forest on wetland', 'Open and re-growing clear-felled, storm-felled or burnt areas on wetlands. Trees are less than 5 meters.', (137, 137, 68)],
    [2, 'Open wetland', 'Open land where the water for a large part of the year is close by, in or just above the ground surface.', (194, 158, 21)],
    [3, 'Arable land', ' Agricultural land used for plant cultivation or kept in such a condition that it can be used for plant cultivation. The land should be able to be used without any special preparatory action other than the use of conventional farming methods and agricultural machinery. The soil can be used for plant cultivation every year. Exceptions can be made for an individual year if special circumstances exist.', (255, 255, 190)],
    [41, 'Non-vegetated other open land', 'Other open land that is not wetland, arable land or exploited vegetation-free surfaces and has less than 10% vegetation coverage during the current vegetation period. The ground can be covered by moss and lichen.', (225, 225, 225)],
    [42, 'Vegetated other open land', 'Other open land that is not wetland, arable land or exploited vegetation-free surfaces and has more than 10% vegetation coverage during the current vegetation period.',  (255, 211, 127)],
    [51, 'Artificial surfaces, building', 'A durable construction consisting of roofs or roofs and walls and which is permanently placed on the ground or partly or wholly below ground or is permanently placed in a certain place in water and is intended to be designed so that people can stay in it.', (90, 20, 20)],
    [52, 'Artificial surfaces, not building or road/railway', 'Artificial open and vegetation-free surfaces that are not building or road/railway.',  (229, 70, 75)],
    [53, 'Artificial surfaces, road/railway', 'Road or railway.',  (25, 25, 25)],
    [61, 'Inland water', 'Lakes or water-courses.',  (102, 153, 205)],
    [62, 'Marine water', 'Sea, ocean, estuaries or coastal lagoons.',  (138, 204, 250)],
    [0, 'Outside mapping area', 'Outside the borders of Sweden and the Exclusive Economic (EEZ) Zone',  (0, 0, 0)]
]


def plot_image_and_mask(img, mask):
    fig, [ax1, ax2, ax3]  = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

    # Get RGB color tuples for the mask and legend, respectively
    img_labels = np.unique(mask)
    all_colors = [[i/255 for i in ele[3]] for ele in land_cover_map_list]
    label_colors = [[i/255 for i in ele[3]] for ele in land_cover_map_list if ele[0] in img_labels]

    # Map from label code in data to it's index in the color map and replace the labels in the
    label_idx = dict(zip([ele[0] for ele in land_cover_map_list if ele[0] in img_labels], np.arange(len(label_colors))))
    idx_mask = np.copy(mask)
    for k, v in label_idx.items(): idx_mask[mask==k] = v

    # Create the color map with a list of the colors present in this mask
    cmap = colors.ListedColormap(label_colors, name='colors')
    
    red=img[:, :, 0]
    green=img[:, :, 1]
    blue=img[:, :, 2]

    # Normalize the colors
    max_value = 255
    brightness = 10
    red = red / max_value * brightness
    green = green / max_value * brightness
    blue = blue / max_value * brightness

    rgb = np.stack([red, green, blue], axis=2)
    
    # Create the legend patches
    patches = [mpatches.Patch(color=all_colors[i], label=land_cover_map_list[i][1] ) for i in range(len(all_colors))]

    ax2.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0. )
    ax1.imshow(rgb)
    ax2.imshow(idx_mask, cmap=cmap)
    ax3.set_visible(False)

    ax1.set_title('RGB sample')
    ax2.set_title('Land cover mask')
    ax3.set_title('Legend for land cover mask')

    return fig


def plot_image_mask_batch(batch):
    '''
    batch is a dict with keys:
        'image': [array of images of shape (13, 256, 256)]
        'mask': [array of masks of shape (1, 256, 256)]
    '''
    def create_image(img):
        '''
        img.shape = (256, 256, 3)
        '''
        red=img[:, :, 0]
        green=img[:, :, 1]
        blue=img[:, :, 2]

        # max_value = np.max([red.max(), green.max(), blue.max()])
        max_value = 255
        brightness = 10
        red = red / max_value * brightness
        green = green / max_value * brightness
        blue = blue / max_value * brightness
        
        rgb_img = np.stack([red, green, blue], axis=2)
        
        return rgb_img

    def create_mask(mask):
        '''
        mask.shape = (256, 256, 1)
        '''
        label_color_map = dict(zip([ele[0] for ele in land_cover_map_list], [ele[3] for ele in land_cover_map_list]))
        rgb_mask = np.array([[[i/255 for i in label_color_map[label]] for label in row] for row in mask])
        return rgb_mask

    images_batch, masks_batch = \
            batch['image'], batch['mask']
    batch_size = len(images_batch)

    fig, axs = plt.subplots(nrows=batch_size, ncols=2, figsize=(8, 4*batch_size))

    for i in range(batch_size):
        image = create_image(images_batch[i, [3, 2, 1], :, :].numpy().transpose((1, 2, 0)))
        mask = create_mask(masks_batch[i, :, :].numpy())
        axs[i][0].imshow(image)
        axs[i][1].imshow(mask)

    # axs[0][0].set_title('RGB samples')
    # axs[0][1].set_title('Land cover masks')
    return fig



if __name__ == '__main__':
    data_dir = 'data\\SentinelLandCoverSweden\\dataset\\'
    file_names = os.listdir(data_dir)
    img = np.genfromtxt(os.path.join(data_dir, file_names[0]), delimiter=',')
    img = img.reshape(256, 256, 14)
    
    # plot_masked_image(img[:, :, :-1], img[:, :, -1], mask_code=61)
    # _m = [[4 for i in j] for j in img[:, :, -1]]
    _m = img[:, :, -1]


    plot_image_and_mask(img[:, :, [3, 2, 1]], _m)




