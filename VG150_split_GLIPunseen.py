import h5py


import json
import numpy as np

with open('DATASET/final_mixed_train_no_coco.json', 'r') as fin: # please refer to https://github.com/microsoft/GLIP for downloading
    res = json.load(fin)
print(res.keys())

GLIP_pretraining_vg_images = set([x['file_name'] for x in res['images'] if x['data_source'] == 'vg'])
print(list(GLIP_pretraining_vg_images)[0])
print(len(GLIP_pretraining_vg_images))


def load_image_filenames(image_file):
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        fns.append(basename)
        img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info

fns, img_info = load_image_filenames("DATASET/VG150/image_data.json")
print(img_info[0])

roidb_file = "DATASET/VG150/VG-SGG-with-attri.h5"
roi_h5 = h5py.File(roidb_file, 'a')
print(roi_h5.keys())

data_split = roi_h5['split'][:]
print((data_split == 2).sum()) # test split
print((data_split == 0).sum()) # train split

import numpy as np
split_GLIPunseen = np.zeros_like(data_split) # no GLIP training samples in test split

test_image_ids = set()
for ind, (info, split) in enumerate(zip(img_info, data_split)):
    if split == 2:
        img_id = info['url'].split('/')[-1].strip()
        test_image_ids.add(img_id)
        if img_id not in GLIP_pretraining_vg_images:
            split_GLIPunseen[ind] = 2 # unseen in GLIP training, as test
        else:
            split_GLIPunseen[ind] = -2 # seen in GLIP training

print(len(test_image_ids))

vg_test_GLIP_unseen = (test_image_ids - GLIP_pretraining_vg_images)

assert (split_GLIPunseen == 2).sum() == len(vg_test_GLIP_unseen)
print(len(vg_test_GLIP_unseen))

if 'split_GLIPunseen' not in list(roi_h5.keys()):
    roi_h5['split_GLIPunseen'] = split_GLIPunseen
roi_h5.close()

# import h5py
# test = h5py.File('test.h5', 'a')
# test = h5py.File('test.h5', 'w')
# test['split_GLIPunseen'] = split_GLIPunseen
# test.close()

# test = h5py.File('test.h5', 'r')
# test['split_GLIPunseen']
# (test['split_GLIPunseen'][:] == 2).sum()