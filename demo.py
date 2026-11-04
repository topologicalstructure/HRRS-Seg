from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation

img_path = 'demo/oem_koeln_50.tif'
label_path = 'demo/oem_koeln_50_label.tif'
img = Image.open(img_path)
label = Image.open(label_path)

name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((896, 896))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = SegEarthSegmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    name_path='./configs/my_name.txt',
    prob_thd=0.1,
    scales_fusion=True,
    sam_parameters=[16, 0.2, 0.5],
    fusion_parameters=[0.125, 0.875, 1, 0.25, 0.875],
    geometric_correction=True,
    scale_normalize=False,
    aggregation_method='vote',
    sub_confi_fusion=False,
    experience_scale=True,
    paint_feature=True,
)

seg_pred = model.predict([img_tensor, img_path], data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis', vmin=0, vmax=len(name_list) - 1)
ax[1].axis('off')
ax[2].imshow(label, cmap='viridis', vmin=0, vmax=len(name_list) - 1)
ax[2].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('seg_pred.png', bbox_inches='tight')