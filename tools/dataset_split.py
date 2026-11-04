import os

data_path = '../data/iSAID/ann_dir/val'
output_path = '../data/iSAID'

files = [f for f in os.listdir(data_path) if not f.startswith('.')]
split_num = len(files) // 2

list1 = []
list2 = []

for counter, filename in enumerate(files):
    base_name = os.path.splitext(filename)[0]
    if base_name.endswith('_instance_color_RGB'):
        base_name = base_name.replace('_instance_color_RGB', '')

    if counter < split_num:
        list1.append(base_name)
    else:
        list2.append(base_name)

with open(os.path.join(output_path, 'val_list1.txt'), 'w') as f:
    for item in list1:
        f.write(item + '\n')

with open(os.path.join(output_path, 'val_list2.txt'), 'w') as f:
    for item in list2:
        f.write(item + '\n')