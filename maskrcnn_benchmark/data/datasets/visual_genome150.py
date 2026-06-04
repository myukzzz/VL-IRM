import os
import copy
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from maskrcnn_benchmark.data.preprocess_utils import object_word_map
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from .od_to_grounding import convert_od_to_grounding_simple, convert_od_to_grounding_with_pseudo_triplet_caption
from .modulated_coco import create_positive_map, create_positive_map_for_od_labels, create_greenlight_map

BOX_SCALE = 1024  # Scale at which we have the boxes
VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
VG150_REL_CATEGORIES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

# Following "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning"
VG150_BASE_OBJ_CATEGORIES = ['__background__', 'tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear']

VG150_BASE_PREDICATE = ['__background__', "between", "to", "made of", "looking at", "along", "laying on", "using", "carrying", "against", "mounted on", "sitting on", "flying in", "covering", "from", "over", "near", "hanging from", "across", "at", "above", "watching", "covered in", "wearing", "holding", "and", "standing on", "lying on", "growing on", "under", "on back of", "with", "has", "in front of", "behind", "parked on"]


VG150_NOVEL2BASE = {'bed': [], 'bench': ['seat'], 'bike': ['vehicle', 'motorcycle'], 'boy': ['men'], 'building': [], 'bus': ['vehicle'], 'car': ['vehicle'], 'cat': ['animal'], 'chair': ['seat'], 'dog': ['animal'], 'door': [], 'ear': [], 'elephant': ['animal'], 'fence': [], 'giraffe': ['animal'], 'girl': ['lady'], 'glass': [], 'hair': [], 'hand': ['paw'], 'hat': ['cap'], 'head': ['men'], 'horse': ['animal'], 'jacket': ['coat'], 'leg': ['branch'], 'man': ['animal'], 'pant': [], 'person': ['people'], 'plane': ['airplane'], 'plate': ['food'], 'pole': [], 'shirt': [], 'shoe': [], 'short': [], 'sign': ['house'], 'snow': [], 'street': ['sidewalk'], 'table': ['board', 'desk'], 'tail': [], 'track': ['railing'], 'train': ['vehicle'], 'tree': ['plant'], 'wheel': ['tire'], 'window': [], 'woman': ['lady'], 'zebra': ['animal']}

#ov-relation
torch.manual_seed(0)
rand_relation_idx = torch.randperm(len(VG150_REL_CATEGORIES) - 1)  # 44 39 7 6 17 29
base_rel_idx = rand_relation_idx[:int((len(VG150_REL_CATEGORIES) - 1) * 0.5)].tolist()
base_rels = []
base_rels.append(VG150_REL_CATEGORIES[0])
for i in base_rel_idx:
    base_rels.append(VG150_REL_CATEGORIES[i + 1])

unseen_rels = [9, 36, 38, 45, 22, 30, 34, 37, 16, 46, 39, 7, 14, 31, 49]
unseen_rels_onlyrel = [unseen_rel -1 for unseen_rel in unseen_rels]

base_rels_new = []
base_rels_new_idx =list(set(rand_relation_idx.tolist())-set(unseen_rels_onlyrel))

base_rels_new.append(VG150_REL_CATEGORIES[0])
for i in base_rels_new_idx:
    base_rels_new.append(VG150_REL_CATEGORIES[i + 1])

base_rels_RAHP = ['at', 'holds', 'wears', 'holding_hands', 'on', 'highfive', 'contain', 'handshake', 'talk_on_phone']
###[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 35, 40, 41, 42, 43, 44, 47, 48, 50]
##########################################OIV6###################################
PREDICATE_BG = '[UNK]'
from . import transforms as T
import pickle
from collections import OrderedDict
from pycocotools.coco import COCO
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
from torchvision.ops.boxes import box_area


def make_coco_transforms(image_set, fix_size=False,
                         strong_aug=False):
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    do_crop =  True

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(data_mean, data_std)
    ])

    # config the params for data aug
    max_size = 1333
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    # scales = getattr(args, 'data_aug_scales', scales)
    # max_size = getattr(args, 'data_aug_max_size', max_size)
    # scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    # scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap =  None
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT

            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        if do_crop:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def OIV6(image_set,disable_transforms=False,tokenizer=None,transforms=None):
    data_path = os.path.join('./DATASET', "oiv6")

    # args=Namespace(config_file='config/GroundingDINO_SwinT_OGC_ovr_oiv6.py',
    #           options={'dn_scalar': 100, 'embed_init_tgt': True, 'dn_label_coef': 1.0, 'dn_bbox_coef': 1.0,
    #                    'use_ema': False, 'dn_box_noise_scale': 1.0, 'max_text_len': 2048, 'batch_size': 1, 'lr': 2e-05,
    #                    'myeval': True, 'myeval_train': True}, dataset_file='sgg_oiv6', data_path='./data',
    #           anno_train_file='oiv6_train_triple.jsonl', anno_val_file='val_bbox_cap.jsonl', coco_panoptic_path=None,
    #           remove_difficult=False, fix_size=False,
    #           output_dir='outputs/ovr_groundingdinot_detection_sgg_oiv6_1231231321321', note='', device='cuda', seed=42,
    #           resume='', pretrain_model_path='./GroundingDINO/weights/groundingdino_swint_ogc.pth',
    #           finetune_ignore=None, start_epoch=0, eval=False, num_workers=8, test=False, debug=False,
    #           find_unused_params=False, half_dtype='bfloat16', save_results=False, save_log=False, world_size=1,
    #           dist_url='env://', rank=0, local_rank=None, amp=False, distributed=False, gpu=0,
    #           modelname='groundingdino', backbone='swin_T_224_1k', position_embedding='sine', pe_temperatureH=20,
    #           pe_temperatureW=20, return_interm_indices=[1, 2, 3], backbone_freeze_keywords=None, enc_layers=6,
    #           dec_layers=6, pre_norm=False, dim_feedforward=2048, hidden_dim=256, dropout=0.0, nheads=8,
    #           num_queries=900, query_dim=4, num_patterns=0, num_feature_levels=4, enc_n_points=4, dec_n_points=4,
    #           two_stage_type='standard', two_stage_bbox_embed_share=False, two_stage_class_embed_share=False,
    #           transformer_activation='relu', dec_pred_bbox_embed_share=True, dn_box_noise_scale=1.0,
    #           dn_label_noise_ratio=0.5, dn_label_coef=1.0, dn_bbox_coef=1.0, embed_init_tgt=True,
    #           dn_labelbook_size=2000, max_text_len=2048, text_encoder_type='bert-base-uncased', use_text_enhancer=True,
    #           use_fusion_layer=True, use_checkpoint=True, use_transformer_ckpt=True, use_text_cross_attention=True,
    #           text_dropout=0.0, fusion_dropout=0.0, fusion_droppath=0.1, sub_sentence_present=True, frozen_weights=None,
    #           frozen_backbone=True, lr=2e-05, lr_rln_mult=10, param_dict_type='default', lr_backbone=1e-05,
    #           lr_text_backbone=1e-05, lr_backbone_names=['backbone.0'],
    #           lr_linear_proj_names=['reference_points', 'sampling_offsets'], lr_linear_proj_mult=0.1,
    #           ddetr_lr_param=False, batch_size=1, weight_decay=0.0001, epochs=12, lr_drop=11,
    #           save_checkpoint_interval=1, clip_max_norm=0.1, onecyclelr=False, multi_step_lr=False,
    #           lr_drop_list=[20, 23], aux_loss=True, set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0,
    #           cls_loss_coef=2.0, mask_loss_coef=1.0, dice_loss_coef=1.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
    #           interm_loss_coef=1.0, no_interm_box_loss=False, focal_alpha=0.25, use_dn=True, dn_number=100, masks=False,
    #           use_text_labels=True, vg_roidb_key='split_GLIPunseen', num_select=300, nms_iou_threshold=0.5,
    #           has_bbox_supervision=True, rln_pretraining=False, do_sgg=True, num_rln_cat=31, num_rln_queries=1,
    #           edge_loss_coef=1.0, rln_add_coords=True, rln_freq_bias=None, sgg_mode='ovr', sg_ovd_mode=False,
    #           sg_ovr_mode=True, focal_loss_for_edges=True, detections_per_img=100, use_distill=True,
    #           unsupervised_distill=True, distill_loss_coef=0.1, eval_before_train=False, dn_scalar=100, use_ema=False,
    #           myeval=True, myeval_train=True)




    strong_aug = False

    # transforms = make_coco_transforms(image_set,
    #                                   fix_size=False,
    #                                   strong_aug=strong_aug) if not disable_transforms else None

    use_text_labels = True
    roidb_key = 'split_GLIPunseen'

    filter_non_overlap = True
    sg_ovd_mode = False
    sg_ovr_mode = True

    return SGGOIV6Dataset(split=image_set,
                          img_dir=os.path.join(data_path, "images"),
                          roidb_file=os.path.join(data_path, "annotations/oiv6"),
                          dict_file=os.path.join(data_path, "stanford_filtered/VG-SGG-dicts.json"),
                          image_file=os.path.join(data_path, "stanford_filtered/image_data.json"),
                          transforms=transforms,
                          use_text_labels=use_text_labels,
                          roidb_key=roidb_key,
                          filter_non_overlap=filter_non_overlap,
                          ovd_mode=sg_ovd_mode,
                          ovr_mode=sg_ovr_mode,
                          use_distill=True,
                          unsupervised_distill=True,
                          gpt4sgg_file=None,
                          use_gpt4sgg=False,
                          tokenizer=tokenizer)



def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

def load_graphs_OIV6(input_boxes,input_relation,box_label,input_filename,
            image_sizes,keep_base_objects=False, keep_base_relations=False,
                ind_to_predicates=None,split=None,base_rels=None,stat_relation=None,
                stat_classes=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    # map to base object classes
    split_mask = np.ones(len(input_boxes)).astype(bool)
    for img_idx in range(len(input_boxes)):
        if keep_base_relations:
            assert split == 'train'
            new_rels = []
            rels=input_relation[img_idx]
            # boxes_all = box_label[img_idx]
            # for box in boxes_all:
            #     stat_classes[box]+=1

            for rel in rels:
                #stat_relation[rel[2]]+=1


                rel_name = ind_to_predicates[rel[2]]
                if rel_name in base_rels:
                        #new_rels.append(rel)
                        ###############VS3
                        new_rels.append([rel[0], rel[1], base_rels.index(rel_name)])

            if len(new_rels) == 0:
                    split_mask[img_idx] = 0
                    continue
            else:
                    rels = np.asarray(new_rels)
                    input_relation[img_idx]=rels

    gt_boxes = [input_boxes[i] for i in np.where(split_mask)[0]]
    relationships = [input_relation[i] for i in np.where(split_mask)[0]]
    gt_classes = [box_label[i] for i in np.where(split_mask)[0]]
    filenames = [input_filename[i] for i in np.where(split_mask)[0]]
    sizes = [image_sizes[i] for i in np.where(split_mask)[0]]

    return gt_boxes, relationships, gt_classes, filenames, sizes#,stat_classes,stat_relation


class SGGOIV6Dataset(torch.utils.data.Dataset):
    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False,
                 num_obj=100,
                 use_text_labels=False,
                 roidb_key='split',
                 ovd_mode=False,
                 ovr_mode=False,
                 use_distill=False,
                 unsupervised_distill=False,
                 gpt4sgg_file=None,
                 use_gpt4sgg=False,
                 tokenizer=None,
                 ):
        """
        Torch dataset (COCO format) for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        assert split in {'train', 'val', 'test'}
        self.num_obj = num_obj
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.roidb_key = roidb_key
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.ovd_mode = ovd_mode
        self.ovr_mode = ovr_mode
        self.use_distill = use_distill
        self.unsupervised_distill = unsupervised_distill
        ######################VS3################
        self.tokenizer=tokenizer
        #########################################


        if self.roidb_key != 'split':
            print("VG dataset use roidb key:", self.roidb_key)

        # ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(
        #     dict_file)  # contiguous 151, 51 containing __background__
        with open(os.path.join(roidb_file, 'oiv6_data_stats.json'), "r") as f:
            merge_data_stats = json.load(f)
        ind_to_classes, ind_to_predicates = merge_data_stats['obj_classes'], merge_data_stats['rel_classes']
        print(ind_to_classes)
        print(ind_to_predicates)

        # replace __background__ with '[bg]'
        if ind_to_predicates[0] == '__background__':
            ind_to_predicates[0] = PREDICATE_BG

        self.ind_to_classes = ind_to_classes
        self.ind_to_predicates = ind_to_predicates

        self.name2classes = OrderedDict(
            {name: idx for idx, name in enumerate(self.ind_to_classes) if name != '__background__'})
        self.class2name = OrderedDict({v: k for k, v in self.name2classes.items()})

        # self.categories = [{'supercategory': 'person',  # not used?
        #                     'id': idx,
        #                     'name': self.ind_to_classes[idx]}
        #                    for idx in range(len(self.ind_to_classes)) if self.ind_to_classes[idx] != '__background__'
        #                    ]


        self.ind_to_predicates = ind_to_predicates
        self.name2predicates = {ind_to_predicates[k]: k for k in range(len(ind_to_predicates))}

        data_file = open(os.path.join(roidb_file, 'oiv6_data_{}.pickle'.format(split)), 'rb')
        merged_data = pickle.load(data_file)

        self.gt_boxes = [data['bboxes'] for data in merged_data]
        self.relationships = [data['relationships'] for data in merged_data]
        self.gt_classes = [data['labels'] for data in merged_data]
        self.filenames = [data['filenames'] for data in merged_data]
        sizes = [data['sizes'] for data in merged_data]

        ##################ov-relation############################
        self.stat_relation = np.zeros((len(self.ind_to_predicates)), dtype=np.int64)
        self.stat_classes = np.zeros((len(self.ind_to_classes)), dtype=np.int64)
        for img_idx in range(len(self.gt_boxes)):
            boxes_all = self.gt_classes[img_idx]
            for box in boxes_all:
                self.stat_classes[box]+=1
            rels = self.relationships[img_idx]
            for rel in rels:
                self.stat_relation[rel[2]]+=1


        # torch.manual_seed(0)
        # rand_relation_idx = torch.randperm(len(self.ind_to_predicates) - 1)  # 14 13 23 27 29 9
        # base_rel_idx = rand_relation_idx[:int((len(self.ind_to_predicates) - 1) * 0.5)].tolist()
        # base_rel_idx = np.argsort(self.stat_relation[1:])[int((len(self.ind_to_predicates) - 1) * 0.5):].tolist()#[26, 24, 4, 18, 21, 9, 13, 20, 28, 8, 16, 1, 7, 2, 0
        # base_rel_idx = np.argsort(self.stat_relation[1:])[::2].tolist()
        base_rels = []
        #base_rels.append(self.ind_to_predicates[0])
        for i in np.argsort(self.stat_relation[1:])[int((len(self.ind_to_predicates) - 1) * 0.5):].tolist():
            base_rels.append(self.ind_to_predicates[i + 1])
        self.base_rels=base_rels
        ##############RAHP############################


        # self.base_rels=base_rels_RAHP

        #OV_RAHP=['surf', 'hang', 'drink', 'ride', 'dance', 'skateboard', 'catch', 'inside_of', 'eat', 'cut', 'kiss', 'interacts_with', 'under', 'hug', 'throw', 'hits', 'snowboard', 'kick', 'ski', 'plays', 'read']

        # 使用列表推导式和enumerate来找索引
        #RAHP_indices = [self.ind_to_predicates.index(x) for x in base_rels_RAHP if x in self.ind_to_predicates]

        #RAHP_stats=np.sort(self.stat_relation[RAHP_indices])

        #aaaaa=0
        ########################################
        self.ind_to_predicates_all = self.ind_to_predicates

        if self.ovr_mode and self.split=='train':
            #self.ind_to_predicates.insert(0, VG150_REL_CATEGORIES[0])
            self.ind_to_predicates=self.base_rels
            self.ind_to_predicates.insert(0, VG150_REL_CATEGORIES[0])
        if self.ovr_mode:
            gt_boxes, relationships, gt_classes, filenames, sizes = load_graphs_OIV6(
                input_boxes=self.gt_boxes,
                input_relation=self.relationships,
                box_label=self.gt_classes,
                input_filename=self.filenames,
                image_sizes=sizes,
                keep_base_objects=(self.split == 'train' and self.ovd_mode),
                keep_base_relations=(self.split == 'train' and self.ovr_mode),
                ind_to_predicates=ind_to_predicates,
                split=self.split,
                base_rels=self.base_rels
            )

            self.gt_boxes = gt_boxes
            self.relationships = relationships
            self.gt_classes = gt_classes
            self.filenames = filenames

        ###########################################

        self.img_info = []
        for i, size in enumerate(sizes):
            image_info = {'image_id': i, 'width': size[0], 'height': size[1]}
            self.img_info.append(image_info)


        # self.keep_nouns = sorted(self.keep_nouns.items(), key=lambda x: x[1], reverse=True)
        # self.keep_rels = sorted(self.keep_rels.items(), key=lambda x: x[1], reverse=True)
        # self.keep_nouns = [(self.ind_to_classes[e[0]], e[1]) for e in self.keep_nouns]
        # self.keep_rels = [(self.ind_to_predicates[e[0]], e[1]) for e in self.keep_rels]
        #
        # self.gt_classes = gt_classes
        # self.relationships = relationships
        # self.gt_boxes = gt_boxes
        # filenames, img_info = load_image_filenames(img_dir, image_file)  # length equals to split_mask
        #
        # filenames = [filenames[i] for i in np.where(split_mask)[0]]
        # img_info = [img_info[i] for i in np.where(split_mask)[0]]
        # assert len(img_info) == len(gt_boxes), " len(img_info) != len(gt_boxes)"

        # GPT4SGG
        self.use_gpt4sgg = use_gpt4sgg and self.split == 'train'
        self.gpt4sgg_data = None
        if self.use_gpt4sgg:
            print("VG150 GPT4SGG :%s used!" % gpt4sgg_file)
            with open(gpt4sgg_file, 'r') as fin:
                self.gpt4sgg_data = json.load(fin)
                self.gpt4sgg_data = {str(e['image_id']): e for e in tqdm(self.gpt4sgg_data)}

        self.images = []
        keep_im = [1] * len(self.filenames)
        for idx, (name, im_info) in enumerate(zip(self.filenames, self.img_info)):
            if self.use_gpt4sgg and str(im_info['image_id']) not in self.gpt4sgg_data:
                keep_im[idx] = 0

            im_info.update({'file_name': name})
            im_info.update({'id': im_info['image_id']})
            self.images.append(im_info)

        self.annotations = []
        self.ids = [im['image_id'] for im, keep in zip(self.images, keep_im) if keep]

        images = []
        rel_list=[]
        for idx, boxes, labels, edges in zip(range(len(self.images)), self.gt_boxes, self.gt_classes,
                                             self.relationships):
            if keep_im[idx] != 1:
                continue
            item = {'image_id': self.images[idx]['image_id'],
                    'boxes': boxes,  # xyxy
                    'labels': labels,
                    'edges': edges,
                    'iscrowd': 0
                    }
            rel_list.append(edges[:,2])
            self.annotations.append(item)
            images.append(self.images[idx])
        max_rel=np.max(np.concatenate(rel_list))
        self.images = images
        self._coco = None

        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once
        ##correct_img_info(self.img_dir, self.image_file)

        self.use_text_labels = use_text_labels

    def categories(self):
        cats_no_bg = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes)) if i > 0} # all categories
        return cats_no_bg


    @property
    def coco(self):
        if self._coco is None:
            _coco = COCO()
            coco_dicts = dict(
                images=self.images,
                annotations=[],
                categories=self.categories)

            for ann in tqdm(self.annotations):
                for cls, box in zip(ann['labels'], ann['boxes']):
                    item = {
                        'area': (box[3] - box[1]) * (box[2] - box[0]),
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # xywh
                        'category_id': cls,
                        'image_id': ann['image_id'],
                        'id': len(coco_dicts['annotations']),
                        'iscrowd': 0,
                    }
                    coco_dicts['annotations'].append(item)

            _coco.dataset = coco_dicts
            _coco.createIndex()
            self._coco = _coco

        return self._coco
    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]
    def __len__(self):
        return len(self.images)

    def _add_pseudo_caption(self, index, target, org_img_size, max_query_len=2048):
        ## pseudo caption from labels
        self.train_text_input_type=''
        if self.split == 'train' and self.train_text_input_type == 'pseudo_triplet_caption':
            annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_with_pseudo_triplet_caption(
                target=target,
                ind_to_classes = self.ind_to_classes,
                ind_to_predicates = self.ind_to_predicates,
            )
        else:######################
            annotations, caption, greenlight_span_for_masked_lm_objective,rel_label_to_positions,predicates_caption  = convert_od_to_grounding_simple(
                target=target,
                image_id=index,
                ind_to_class={i: n for i, n in enumerate(self.ind_to_classes) if i > 0},
                ind_to_predicates={i: n for i, n in enumerate(self.ind_to_predicates) if i > 0},
                separation_tokens='. ',
            )

        # image info
        target.add_field("image_id", index)
        target.add_field("area", torch.tensor([obj["area"] for obj in annotations]))
        target.add_field("iscrowd", torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations]))
        w, h = org_img_size
        target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
        target.add_field("size", torch.as_tensor([int(h), int(w)]))

        # pseudo caption
        tokens_positive = [obj["tokens_positive"] for obj in annotations]
        target.add_field("caption", caption)

        target.add_field("rel_label_to_positions", rel_label_to_positions)
        target.add_field("caption_rel", predicates_caption)

        target.add_field("tokens_positive", tokens_positive)
        tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
        target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))###positive_map-每个box对应的token坐标
        target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))###greenlight_map——整个图上所有box对应的token坐标
        target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))#没意义

        return target



    def __getitem__(self, index):
        item = self.images[index]
        image_id = item["image_id"]
        w, h = item['width'], item['height']

        # load image
        img = Image.open(os.path.join(self.img_dir, os.path.basename(item['file_name']))).convert("RGB")
        if img.size[0] != item['width'] or img.size[1] != item['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ',
                  str(item['width']), ' ', str(item['height']), ' ', '=' * 20)

        # flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')#########false
        #
        # # load gt
        # gt_boxes, gt_classes, gt_edges = self.get_groundtruth(index)
        #
        # if self.filter_duplicate_rels and self.split == "train":
        #     # Filter out dupes!
        #     old_size = gt_edges.shape[0]
        #     all_rel_sets = defaultdict(list)
        #     for (o0, o1, r) in gt_edges:
        #         all_rel_sets[(o0, o1)].append(r)
        #     gt_edges = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
        #     gt_edges = np.array(gt_edges, dtype=np.int32)
        #
        # if self.split == "train":
        #     gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(None, h)
        #     gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(None, w)
        #
        # if flip_img:
        #     new_xmin = w - gt_boxes[:, 2]
        #     new_xmax = w - gt_boxes[:, 0]
        #     gt_boxes[:, 0] = new_xmin
        #     gt_boxes[:, 2] = new_xmax
        #
        #     img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        #
        # target = dict()
        # target["iscrowd"] = torch.zeros(gt_boxes.shape[0], dtype=torch.float32)
        # target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float32)
        # target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        # target["edges"] = torch.as_tensor(gt_edges, dtype=torch.int64)
        # target["image_id"] = image_id
        # target["orig_size"] = torch.as_tensor([int(h), int(w)])
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        ###################VS3#############################

        target_VS3 = self.get_groundtruth(index)
        org_img_size = img.size
        if self.transforms is not None:
            img, target_VS3 = self.transforms(img, target_VS3)

        target_VS3 = self._add_pseudo_caption(index, target_VS3, org_img_size)#######插入positive map
        target_VS3.add_field('filename', self.filenames[index])
        target_VS3.add_field('all_rels', self.ind_to_predicates_all)

        ###############################################




        # text
        # if self.use_text_labels:
        #     labels = target['labels'].cpu().numpy().tolist()
        #     gt_sample = [self.ind_to_classes[e] for e in labels]
        #
        #     triples = []
        #     nouns = []
        #     rels = []
        #     for edge in target['edges']:
        #         sub = gt_sample[edge[0]]
        #         obj = gt_sample[edge[1]]
        #
        #         triple = (sub,
        #                   obj,
        #                   self.ind_to_predicates[edge[2]])
        #         triples.append(triple)
        #         nouns.append(sub)
        #         nouns.append(obj)
        #         rels.append(triple[2])
        #
        #     target['relations'] = triples
        #     nouns = list(set(nouns))
        #     rels = list(set(rels))
        #
        #     target['gt_names'] = copy.deepcopy(gt_sample)
        #
        #     all_nouns = self.ind_to_classes[1:] if (
        #             self.split == 'train' and self.ovd_mode) else self.ind_to_classes[1:]
        #
        #     # if self.ovd_mode and self.ovr_mode: # release the constraint
        #     #    all_nouns = VG150_OBJ_CATEGORIES[1:]
        #     if self.split == 'train':
        #         random.shuffle(all_nouns)
        #
        #     target['caption'] = '. '.join(all_nouns)
        #     target['caption'] = preprocess_caption(target['caption'])
        #
        #     target['gt_rels'] = copy.deepcopy(rels)
        #     all_rels = self.ind_to_predicates[1:]  # include __background__
        #     if self.split == 'train':
        #         random.shuffle(all_rels)
        #
        #     rel_caption = '. '.join(all_rels) + '.'
        #     target['rel_caption'] = rel_caption
        #
        # if len(target['edges']) == 0 and self.split == 'train':
        #     return self[index - random.randint(1, 10)]  # remap to a valid sample


        return img, target_VS3,index

    # def get_groundtruth(self, index):
    #     item = self.images[index]
    #     image_id = item["image_id"]
    #     w, h = item['width'], item['height']
    #
    #     ann = self.annotations[index]
    #     gt_boxes = ann['boxes']
    #     gt_classes = ann['labels']
    #     gt_edges = ann['edges']
    #
    #     # truncate the gt if need
    #     max_obj = self.num_obj
    #     if len(gt_boxes) > max_obj:
    #         idx = np.random.choice(torch.arange(len(gt_boxes)), max_obj, replace=False)
    #         gt_boxes = gt_boxes[idx, :]
    #         gt_classes = gt_classes[idx]
    #         gt_edges = [[idx.tolist().index(rel[0]), idx.tolist().index(rel[1]), rel[2]] for rel in gt_edges if
    #                     rel[0] in idx and rel[1] in idx]
    #         gt_edges = np.array(gt_edges)
    #
    #     gt_boxes = torch.from_numpy(np.asarray(gt_boxes, dtype=float)).reshape(-1, 4)  # guard against no boxes
    #
    #     gt_classes = torch.as_tensor(gt_classes, dtype=torch.int64)
    #
    #     #         combine entities by with iou>0.9
    #     all_entities_infos_after_iou_merge = []
    #     ent_match_info = (box_iou(gt_boxes, gt_boxes)[0] > 0.9) & (gt_classes.unsqueeze(1) == gt_classes)
    #
    #     keep_tgt_ids, ent_id_to_new_id = [], {}
    #     for ent_id in range(len(ent_match_info)):
    #         matched_ent_ids = ent_match_info[ent_id].nonzero().squeeze(1).tolist()
    #         if len(matched_ent_ids) > 0:
    #             keep_tgt_ids.append(ent_id)
    #             for id in matched_ent_ids: ent_id_to_new_id[id] = len(all_entities_infos_after_iou_merge)
    #             # all_entities_infos_after_iou_merge.append(all_entities_infos[ent_id])
    #             all_entities_infos_after_iou_merge.append(-1)
    #             ent_match_info[:, matched_ent_ids] = False
    #
    #     gt_boxes = gt_boxes[keep_tgt_ids]
    #     gt_classes = gt_classes[keep_tgt_ids]
    #     new_edges = []
    #     for edge in gt_edges:
    #         new_edges.append([ent_id_to_new_id[edge[0]], ent_id_to_new_id[edge[1]], edge[2]])
    #     gt_edges = np.asarray(new_edges)
    #
    #     return gt_boxes, gt_classes, gt_edges


    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        # use grounded boxes for training


        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        #target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target


    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(self, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            # 'att_classes': self.ind_to_attributes,
        }
        return result

#################################################################################

class VG150Dataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000, tokenizer=None, split_key='split', train_text_input_type='obj_categories',
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, box_grounding_file=None, open_vocabulary_mode=False,train_ov_relation=False,more_base_rel=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # # for debug
        # num_im = 100
        # num_val_im = 0

        assert split in {'train', 'val', 'test'}
        self.train_text_input_type = train_text_input_type
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.open_vocabulary_mode = open_vocabulary_mode
        self.train_ov_relation = train_ov_relation

        # train with grounded unlocalized scene graphs
        self.train_with_grounding_boxes = False
        if self.split == 'train' and box_grounding_file is not None:
            self.train_with_grounding_boxes = True
            with open(box_grounding_file, 'r') as fin:
                self.grounded_boxes = json.load(fin)

        # load annotations
        self.ind_to_classes, self.ind_to_predicates_all, self.ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__
        if (self.split=='train' and self.open_vocabulary_mode): self.ind_to_classes = VG150_BASE_OBJ_CATEGORIES


        if (self.split == 'train' and self.train_ov_relation):
            if more_base_rel:
                self.ind_to_predicates = base_rels_new
            else:
                self.ind_to_predicates = base_rels
        else:
            _, self.ind_to_predicates, _ = load_info(dict_file)



        self.categories_with_bg = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
            split_key=split_key,
            keep_base_classes=(self.split=='train' and self.open_vocabulary_mode),
            keep_base_relation = (self.split == 'train' and self.train_ov_relation),
            more_base_rel=more_base_rel
        )

        self.filenames, self.img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

    def categories(self):
        cats_no_bg = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes)) if i > 0} # all categories
        return cats_no_bg

    def __getitem__(self, index):
        #if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        img = Image.open(self.filenames[index]).convert("RGB")
        org_img_size = img.size
        if org_img_size[0] != self.img_info[index]['width'] or org_img_size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(org_img_size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        target = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = self._add_pseudo_caption(index, target, org_img_size)#######插入positive map
        target.add_field('filename', self.filenames[index])
        target.add_field('all_rels', self.ind_to_predicates_all)

        return img, target, index

    def _add_pseudo_caption(self, index, target, org_img_size, max_query_len=310):
        ## pseudo caption from labels
        if self.split == 'train' and self.train_text_input_type == 'pseudo_triplet_caption':
            annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_with_pseudo_triplet_caption(
                target=target,
                ind_to_classes = self.ind_to_classes,
                ind_to_predicates = self.ind_to_predicates,
            )
        else:######################
            annotations, caption, greenlight_span_for_masked_lm_objective,rel_label_to_positions,predicates_caption  = convert_od_to_grounding_simple(
                target=target,
                image_id=index,
                ind_to_class={i: n for i, n in enumerate(self.ind_to_classes) if i > 0},
                ind_to_predicates={i: n for i, n in enumerate(self.ind_to_predicates) if i > 0},
                separation_tokens='. ',
            )

        # image info
        target.add_field("image_id", index)
        target.add_field("area", torch.tensor([obj["area"] for obj in annotations]))
        target.add_field("iscrowd", torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations]))
        w, h = org_img_size
        target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
        target.add_field("size", torch.as_tensor([int(h), int(w)]))

        # pseudo caption
        tokens_positive = [obj["tokens_positive"] for obj in annotations]
        target.add_field("caption", caption)

        target.add_field("rel_label_to_positions", rel_label_to_positions)
        target.add_field("caption_rel", predicates_caption)

        target.add_field("tokens_positive", tokens_positive)
        tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
        target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))###positive_map-每个box对应的token坐标
        target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))###greenlight_map——整个图上所有box对应的token坐标
        target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))#没意义

        return target

    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.dict_file,
                                                image_file=self.image_file, must_overlap=True)
        fg_matrix[:, :, 0] = bg_matrix
        fg_matrix = np.log(fg_matrix + 2) # smooth frequency
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None])

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        # use grounded boxes for training
        if self.split == 'train' and self.train_with_grounding_boxes:
            box = torch.zeros_like(box)
            box[:, 2] = w-1; box[:, 3] = h-1
            for inst_id, box_info in self.grounded_boxes[str(img_info['image_id'])].items():
                box[int(inst_id)] = torch.tensor(box_info['bbox'])

        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:##################
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.filenames)


def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    train_data = VG150Dataset(split='train', img_dir=img_dir, roidb_file=roidb_file,
                        dict_file=dict_file, image_file=image_file, num_val_im=5000,
                        filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, split_key='split', keep_base_classes=False,keep_base_relation=False,more_base_rel=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5[split_key][:]####按GLIP未见过的类进行data分割
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':#####5000之后为train
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:########################
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        # map to base classes——oV-检测实体OV——map+去掉一部分无法map的数据
        if keep_base_classes:##########################
            assert split == 'train'
            new_boxes, new_labels, new_attrs, org_bid2new = [], [], [], {}
            for org_bid, (b, l, a) in enumerate(zip(boxes_i, gt_classes_i, gt_attributes_i)):
                org_label = VG150_OBJ_CATEGORIES[l]

                mapped_l = None
                if org_label in VG150_BASE_OBJ_CATEGORIES:
                    mapped_l = VG150_BASE_OBJ_CATEGORIES.index(org_label)
                # elif len(VG150_NOVEL2BASE[org_label]) > 0:#novel2base映射或删去novel样本
                #     mapped_l = VG150_BASE_OBJ_CATEGORIES.index(random.choice(VG150_NOVEL2BASE[org_label]))

                if mapped_l is not None:######解决删数据错位问题
                    org_bid2new[org_bid] = len(new_boxes)
                    new_boxes.append(b)
                    new_labels.append(mapped_l)
                    new_attrs.append(a)#dim=10

            # filter relations
            new_rels = []
            for r in rels:
                if r[0] in org_bid2new and r[1] in org_bid2new:
                    new_rels.append([org_bid2new[r[0]], org_bid2new[r[1]], r[2]])#改变三元组idx



            changed_rels=new_rels



            if len(changed_rels) == 0 or len(new_boxes) < 2:
                split_mask[image_index[i]] = 0
                continue
            else:
                boxes_i = np.stack(new_boxes)
                gt_classes_i = np.array(new_labels)
                gt_attributes_i = np.stack(new_attrs)
                rels = np.array(changed_rels)
##########################################################################

        if keep_base_relation:
                    assert split == 'train'


                    ################################################################################################
                    new_relations, org_bid2new_rel = [], {}



                    for idx, r in enumerate(rels):
                        org_rel_label = VG150_REL_CATEGORIES[r[2]]
                        mapped_rel_l = None
                        if more_base_rel:
                            if org_rel_label in base_rels_new:
                                mapped_rel_l = base_rels_new.index(org_rel_label)
                        else:
                            if org_rel_label in base_rels:
                                mapped_rel_l = base_rels.index(org_rel_label)
                            #mapped_rel_l = r[2]

                        if mapped_rel_l is not None:
                            org_bid2new_rel[idx] = len(new_relations)
                            new_relations.append([r[0], r[1], mapped_rel_l])

                    # if len(new_relations) == 0 or len(boxes_i) < 2:
                    #     split_mask[image_index[i]] = 0
                    #     continue
                    # else:###########################没调好


                    rels = np.array(new_relations)



        if filter_non_overlap:#false
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships

# class COCOCaptionSceneGraphDatasetV1(torch.utils.data.Dataset):
#     def __init__(self, img_dir, img_meta_info_file, caption_scene_graph_file, transforms=None, tokenizer=None, train_text_input_type='obj_categories'):
#         self.train_text_input_type = train_text_input_type
#         self.img_dir = img_dir
#         self.transforms = transforms
#         self.ind_to_classes = VG150_OBJ_CATEGORIES
#         self.ind_to_predicates = VG150_REL_CATEGORIES
#         self.tokenizer = tokenizer
#
#         # parsed scene graph infos
#         with open(caption_scene_graph_file, 'r') as fin:
#             self.caption_sg_infos = json.load(fin)
#         grounded_img_ids = self.caption_sg_infos['sg_grounding_infos']['grounded_img_ids']
#         vg150_valid_img_ids = [str(x) for x in self.caption_sg_infos['vg150_valid_infos']['vg150_valid_image_ids']]
#         self.valid_img_ids = list(set(grounded_img_ids) & set(vg150_valid_img_ids))
#
#         # orginal image meta infos
#         with open(img_meta_info_file, 'r') as fin:
#             self.org_caption_infos = json.load(fin)
#         self.img_info = {str(x['id']): x for x in self.org_caption_infos['images']}
#
#     def __getitem__(self, index):
#         img_id = self.valid_img_ids[index]
#         img_filename = f"{self.img_dir}/{self.img_info[img_id]['file_name']}"
#         img = Image.open(img_filename).convert("RGB")
#         org_img_size = img.size
#         if org_img_size[0] != self.img_info[img_id]['width'] or org_img_size[1] != self.img_info[img_id]['height']:
#             print('='*20, ' ERROR index ', str(index), ' ', str(org_img_size), ' ', str(self.img_info[img_id]['width']), ' ', str(self.img_info[img_id]['height']), ' ', '='*20)
#
#         target, caption, all_entities_infos = self.get_groundtruth(index)
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#
#         self._add_extra_infos(index, target, org_img_size, caption, all_entities_infos)
#         return img, target, index
#
#     def get_img_info(self, index):
#         img_id = self.valid_img_ids[index]
#         return self.img_info[img_id]
#
#     def get_groundtruth(self, index):
#         img_id = self.valid_img_ids[index]
#         img_cap_with_sgs = [x for x in self.caption_sg_infos['img_captions_with_parsed_graph'][img_id] if x['vg150_valid']]
#
#         ############# collect info from all captions #############
#         all_captions, all_entities_infos, all_relations = [], [], []
#         random.shuffle(img_cap_with_sgs) # augmentation
#         ent_orgid_to_merged_id = {}; ent_char_span_offset = 0
#         for cap_id, cap in enumerate(img_cap_with_sgs):
#             all_captions.append(cap['caption'])
#             # entities
#             for orgid, ent in enumerate(copy.deepcopy(cap['text_scene_graph']['entities'])):
#                 if len(ent['vg150_obj_category']) > 0:
#                     ent_orgid_to_merged_id[f"{str(cap_id)}-{str(orgid)}"] = len(all_entities_infos)
#                     ent['char_span'] = [ent['char_span'][0]+ent_char_span_offset, ent['char_span'][1]+ent_char_span_offset]
#                     all_entities_infos.append(ent)
#             # relation instances
#             for rel in copy.deepcopy(cap['text_scene_graph']['relations']): # !! avoid changing original values
#                 subj_orgid, obj_orgid = f"{str(cap_id)}-{str(rel['subject'])}", f"{str(cap_id)}-{str(rel['object'])}"
#                 if rel['vg150_valid']:
#                     rel['subject'] = ent_orgid_to_merged_id[subj_orgid]
#                     rel['object'] = ent_orgid_to_merged_id[obj_orgid]
#                     all_relations.append(rel)
#             ent_char_span_offset += len(cap['caption'])+2
#             # break # use triplets from single image
#         caption = ". ".join(all_captions)
#
#         ############# create target box ####################
#         h, w = self.img_info[img_id]['height'], self.img_info[img_id]['width']
#         box = torch.zeros((len(all_entities_infos), 4)); box[:, 2] = w-1; box[:, 3] = h-1
#         box_labels = []
#         for ent_id, ent_info in enumerate(all_entities_infos):
#             if 'ground_box' in ent_info:
#                 box[ent_id] = torch.tensor(ent_info['ground_box'])
#             label_name = random.choice(ent_info['vg150_obj_category'])
#             box_labels.append(VG150_OBJ_CATEGORIES.index(label_name))
#
#         target = BoxList(box, (w, h), 'xyxy') # xyxy
#         target.add_field('labels', torch.tensor(box_labels))
#
#         # combine entities by with iou>0.9
#         all_entities_infos_after_iou_merge = []
#         ent_match_info = (boxlist_iou(target, target) > 0.9) & (target.get_field('labels').unsqueeze(1) == target.get_field('labels').unsqueeze(0))
#         keep_tgt_ids, ent_id_to_new_id = [], {}
#         for ent_id in range(len(ent_match_info)):
#             matched_ent_ids = ent_match_info[ent_id].nonzero().squeeze(1).tolist()
#             if len(matched_ent_ids) > 0:
#                 keep_tgt_ids.append(ent_id)
#                 for id in matched_ent_ids: ent_id_to_new_id[id] = len(all_entities_infos_after_iou_merge)
#                 all_entities_infos_after_iou_merge.append(all_entities_infos[ent_id])
#                 ent_match_info[:, matched_ent_ids] = False
#         target = target[keep_tgt_ids]
#         for rel in all_relations:
#             rel['subject'] = ent_id_to_new_id[rel['subject']]
#             rel['object'] = ent_id_to_new_id[rel['object']]
#
#         ############# create target relations ##############
#         relation_map = torch.zeros((len(target), len(target)), dtype=torch.int64)
#         for rel in all_relations:
#             rel_label_name = random.choice(rel['vg150_rel_category'])
#             if relation_map[rel['subject'], rel['object']] > 0:
#                 if random.random() > 0.5:
#                     relation_map[rel['subject'], rel['object']] = VG150_REL_CATEGORIES.index(rel_label_name)
#             else:
#                 relation_map[rel['subject'], rel['object']] = VG150_REL_CATEGORIES.index(rel_label_name)
#         target.add_field("relation", relation_map)
#
#         target = target.clip_to_image(remove_empty=True)
#         return target, caption, all_entities_infos_after_iou_merge
#
#     def _add_extra_infos(self, index, target, org_img_size, caption, all_entities_infos, max_query_len=310):
#         w, h = org_img_size
#         ############# add extra info, e.g. text input ##############
#         target.add_field("image_id", index)
#         target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
#         target.add_field("size", torch.as_tensor([int(h), int(w)]))
#
#         # caption todo: check
#         target.add_field("org_caption", caption)
#         if self.train_text_input_type == 'natural_caption':
#             tokens_positive = [[obj["char_span"]] for obj in all_entities_infos]
#         elif self.train_text_input_type == 'pseudo_triplet_caption':
#             annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_with_pseudo_triplet_caption(
#                 target=target,
#                 ind_to_classes = self.ind_to_classes,
#                 ind_to_predicates = self.ind_to_predicates,
#             )
#             tokens_positive = [obj["tokens_positive"] for obj in annotations]
#         else: # obj_categories
#             annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_simple(
#                 target=target,
#                 image_id=index,
#                 ind_to_class={i: n for i, n in enumerate(self.ind_to_classes) if i > 0},
#                 separation_tokens='. ',
#             )
#             tokens_positive = [obj["tokens_positive"] for obj in annotations]
#
#         target.add_field("caption", caption)
#         target.add_field("tokens_positive", tokens_positive)
#         tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
#         target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))
#         target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))
#         target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))
#         return target
#
#     def __len__(self):
#         return len(self.valid_img_ids)

# Only used as training dataset
class VGCaptionSceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_meta_info_file, caption_scene_graph_file, transforms=None, tokenizer=None, train_text_input_type='obj_categories', open_vocabulary_mode=False):
        self.train_text_input_type = train_text_input_type
        self.img_dir = img_dir
        self.transforms = transforms
        self.ind_to_classes = VG150_OBJ_CATEGORIES
        self.ind_to_predicates = VG150_REL_CATEGORIES
        self.tokenizer = tokenizer

        self.open_vocabulary_mode = open_vocabulary_mode
        if self.open_vocabulary_mode:
            self.ind_to_classes = VG150_BASE_OBJ_CATEGORIES

        # parsed scene graph infos
        with open(caption_scene_graph_file, 'r', encoding='utf-8') as fin:
            self.caption_sg_infos = json.load(fin)
        grounded_img_ids = [str(id) for id in self.caption_sg_infos['sg_grounding_infos']['grounded_img_ids']]

        with_rel_img_ids = [k for k, v in self.caption_sg_infos['text_scene_graph'].items() if len(v['triplets']) > 0 and len(v['objects']) > 1]
        self.valid_img_ids = list(set(grounded_img_ids) & set(with_rel_img_ids))

        self._load_all_image_infos(img_meta_info_file)

    def _load_all_image_infos(self, img_meta_info_file):
        # orginal image meta infos
        with open(img_meta_info_file, 'r') as fin:
            self.org_caption_infos = json.load(fin)
        self.img_info = {str(x['image_id']): x for x in self.org_caption_infos}

    def __getitem__(self, index):
        img_id = self.valid_img_ids[index]
        img_filename = f"{self.img_dir}/{str(img_id)}.jpg"
        img = Image.open(img_filename).convert("RGB")
        org_img_size = img.size
        if org_img_size[0] != self.img_info[img_id]['width'] or org_img_size[1] != self.img_info[img_id]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(org_img_size), ' ', str(self.img_info[img_id]['width']), ' ', str(self.img_info[img_id]['height']), ' ', '='*20)

        target = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        self._add_extra_infos(index, target, org_img_size)

        # at least 2 objects and 1 relation
        if len(target) >= 2 and len(target.get_field('relation').nonzero()) >= 1:
            return img, target, index
        else:
            return self[index - random.randint(1,10)] # remap to a valid sample

    def get_img_info(self, index):
        img_id = self.valid_img_ids[index]
        return self.img_info[img_id]

    def get_groundtruth(self, index):
        img_id = self.valid_img_ids[index]

        img_caption_sg = self.caption_sg_infos['text_scene_graph'][str(img_id)]
        all_entities_infos = img_caption_sg["entities_after_grounding"]
        all_relations = img_caption_sg["relations_after_grounding"]

        ############# create target box ####################
        h, w = self.img_info[img_id]['height'], self.img_info[img_id]['width']
        box = torch.zeros((len(all_entities_infos), 4)); box[:, 2] = w-1; box[:, 3] = h-1
        box_labels = []
        for ent_id, ent_info in enumerate(all_entities_infos):
            box[ent_id] = torch.tensor(ent_info['xyxy_box'])
            box_labels.append(ent_info['vg150_obj_label'])

        # map to base classes
        if self.open_vocabulary_mode:
            box, box_labels, all_entities_infos, all_relations = self._map_to_base_classes(box, box_labels, all_entities_infos, all_relations)

        target = BoxList(box, (w, h), 'xyxy') # xyxy
        target.add_field('labels', torch.tensor(box_labels))

        # combine entities by with iou>0.9
        all_entities_infos_after_iou_merge = []
        ent_match_info = (boxlist_iou(target, target) > 0.9) & (target.get_field('labels').unsqueeze(1) == target.get_field('labels').unsqueeze(0))
        keep_tgt_ids, ent_id_to_new_id = [], {}
        for ent_id in range(len(ent_match_info)):
            matched_ent_ids = ent_match_info[ent_id].nonzero().squeeze(1).tolist()
            if len(matched_ent_ids) > 0:
                keep_tgt_ids.append(ent_id)
                for id in matched_ent_ids: ent_id_to_new_id[id] = len(all_entities_infos_after_iou_merge)
                all_entities_infos_after_iou_merge.append(all_entities_infos[ent_id])
                ent_match_info[:, matched_ent_ids] = False
        target = target[keep_tgt_ids]
        for rel in all_relations:
            rel['subject'] = ent_id_to_new_id[rel['subject']]
            rel['object'] = ent_id_to_new_id[rel['object']]

        ############# create target relations ##############
        relation_map = torch.zeros((len(target), len(target)), dtype=torch.int64)
        for rel in all_relations:
            if rel['subject'] == rel['object']: continue
            if relation_map[rel['subject'], rel['object']] > 0:
                if random.random() > 0.5:
                    relation_map[rel['subject'], rel['object']] = rel['vg150_predicate_label']
            else:
                relation_map[rel['subject'], rel['object']] = rel['vg150_predicate_label']
        target.add_field("relation", relation_map)

        target = target.clip_to_image(remove_empty=True)
        return target

    def _map_to_base_classes(self, box, box_labels, all_entities_infos, all_relations):
        new_boxes, new_labels, new_all_entities_infos, org_bid2new = [], [], [], {}
        for org_bid, (b, l, ent_info) in enumerate(zip(box, box_labels, all_entities_infos)):
            org_label = VG150_OBJ_CATEGORIES[l]

            mapped_l = None
            if org_label in VG150_BASE_OBJ_CATEGORIES:
                mapped_l = VG150_BASE_OBJ_CATEGORIES.index(org_label)
            elif len(VG150_NOVEL2BASE[org_label]) > 0:
                mapped_l = VG150_BASE_OBJ_CATEGORIES.index(random.choice(VG150_NOVEL2BASE[org_label]))

            if mapped_l is not None:
                org_bid2new[org_bid] = len(new_boxes)
                new_boxes.append(b)
                new_labels.append(mapped_l)
                new_all_entities_infos.append(ent_info)

        if len(new_boxes) > 0:
            new_boxes = torch.stack(new_boxes, dim=0)
        else:
            new_boxes = torch.zeros((0,4))

        # filter relations
        new_rels = []
        for r in all_relations:
            if r['subject'] in org_bid2new and r['object'] in org_bid2new:
                new_rels.append({
                    'subject': org_bid2new[r['subject']],
                    'object': org_bid2new[r['object']],
                    'vg150_predicate_label': r['vg150_predicate_label']
                })

        return new_boxes, new_labels, new_all_entities_infos, new_rels

    def _add_extra_infos(self, index, target, org_img_size, max_query_len=310):
        w, h = org_img_size
        ############# add extra info, e.g. text input ##############
        target.add_field("image_id", index)
        target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
        target.add_field("size", torch.as_tensor([int(h), int(w)]))

        # caption
        if self.train_text_input_type == 'pseudo_triplet_caption':
            annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_with_pseudo_triplet_caption(
                target=target,
                ind_to_classes = self.ind_to_classes,
                ind_to_predicates = self.ind_to_predicates,
            )
            tokens_positive = [obj["tokens_positive"] for obj in annotations]
        else: # obj_categories
            annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_simple(
                target=target,
                image_id=index,
                ind_to_class={i: n for i, n in enumerate(self.ind_to_classes) if i > 0},
                separation_tokens='. ',
            )
            tokens_positive = [obj["tokens_positive"] for obj in annotations]

        target.add_field("caption", caption)
        target.add_field("tokens_positive", tokens_positive)
        tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
        target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))
        target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))
        target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))
        return target

    def __len__(self):
        return len(self.valid_img_ids)

class COCOCaptionSceneGraphDataset(VGCaptionSceneGraphDataset):
    def __init__(self, img_dir, img_meta_info_file, caption_scene_graph_file, transforms=None, tokenizer=None, train_text_input_type='obj_categories', open_vocabulary_mode=False):
        super(COCOCaptionSceneGraphDataset, self).__init__(img_dir, img_meta_info_file, caption_scene_graph_file, transforms, tokenizer, train_text_input_type, open_vocabulary_mode)

    def _load_all_image_infos(self, img_meta_info_file):
        # orginal image meta infos
        with open(img_meta_info_file, 'r') as fin:
            self.org_caption_infos = json.load(fin)
        self.img_info = {str(x['id']): x for x in self.org_caption_infos['images']}

    def __getitem__(self, index):
        img_id = self.valid_img_ids[index]
        img_filename = f"{self.img_dir}/{self.img_info[img_id]['file_name']}"
        img = Image.open(img_filename).convert("RGB")
        org_img_size = img.size
        if org_img_size[0] != self.img_info[img_id]['width'] or org_img_size[1] != self.img_info[img_id]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(org_img_size), ' ', str(self.img_info[img_id]['width']), ' ', str(self.img_info[img_id]['height']), ' ', '='*20)

        target = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        self._add_extra_infos(index, target, org_img_size)

        # at least 2 objects and 1 relation
        if len(target) >= 2 and len(target.get_field('relation').nonzero()) >= 1:
            return img, target, index
        else:
            return self[index - random.randint(1,10)] # resample a valid one

class UnboundedVGSceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_meta_info_file, scene_graph_file, transforms=None, tokenizer=None, train_text_input_type='obj_categories'):
        self.train_text_input_type = train_text_input_type
        self.img_dir = img_dir
        self.transforms = transforms
        self.ind_to_classes = VG150_OBJ_CATEGORIES
        self.ind_to_predicates = VG150_REL_CATEGORIES
        self.tokenizer = tokenizer

        # image-level scene graph infos
        with open(scene_graph_file, 'r', encoding='utf-8') as fin:
            self.sg_infos = json.load(fin)
        self.valid_img_ids = list(self.sg_infos.keys())

        self._load_all_image_infos(img_meta_info_file)

    def _load_all_image_infos(self, img_meta_info_file):
        # orginal image meta infos
        with open(img_meta_info_file, 'r') as fin:
            self.org_caption_infos = json.load(fin)
        self.img_info = {str(x['image_id']): x for x in self.org_caption_infos}

    def get_img_info(self, index):
        img_id = self.valid_img_ids[index]
        return self.img_info[img_id]

    def __getitem__(self, index):
        img_id = self.valid_img_ids[index]
        img_filename = f"{self.img_dir}/{str(img_id)}.jpg"
        img = Image.open(img_filename).convert("RGB")
        org_img_size = img.size
        if org_img_size[0] != self.img_info[img_id]['width'] or org_img_size[1] != self.img_info[img_id]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(org_img_size), ' ', str(self.img_info[img_id]['width']), ' ', str(self.img_info[img_id]['height']), ' ', '='*20)

        target, box_labels, relation_labels_dict = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        self._add_extra_infos(index, target, org_img_size, box_labels, relation_labels_dict)

        return img, target, index

    def __len__(self):
        return len(self.valid_img_ids)

    def get_groundtruth(self, index):
        img_id = self.valid_img_ids[index]
        img_caption_sg = self.sg_infos[str(img_id)]

        ############# create target box ####################
        h, w = self.img_info[img_id]['height'], self.img_info[img_id]['width']
        box = torch.zeros((len(img_caption_sg["objects"]), 4)); box[:, 2] = w-1; box[:, 3] = h-1
        box_labels = []
        for ent_id, ent_info in enumerate(img_caption_sg["objects"]):
            box[ent_id] = torch.tensor([ent_info['x'], ent_info['y'], ent_info['x']+ent_info['w']-1, ent_info['y']+ent_info['h']-1])
            box_labels.append(random.choice(ent_info['names']).lower().strip())
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        ############# create target relations ##############
        relation_labels_dict = {}
        relation_map = torch.zeros((len(target), len(target)), dtype=torch.int64)
        random.shuffle(img_caption_sg["relations"])
        for rel in img_caption_sg["relations"]:
            relation_map[rel['subject_ind'], rel['object_ind']] = 1
            relation_labels_dict[(rel['subject_ind'], rel['object_ind'])] = rel['predicate'].strip()
        target.add_field("relation", relation_map)

        target = target.clip_to_image(remove_empty=False)
        return target, box_labels, relation_labels_dict

    def _add_extra_infos(self, index, target, org_img_size, box_labels, relation_labels_dict, max_query_len=310):
        w, h = org_img_size
        ############# add extra info, e.g. text input ##############
        target.add_field("image_id", index)
        target.add_field("orig_size", torch.as_tensor([int(h), int(w)]))
        target.add_field("size", torch.as_tensor([int(h), int(w)]))
        target.add_field('box_labels', box_labels)
        target.add_field('relation_labels_dict', relation_labels_dict)

        # compose caption and grounding infos
        annotations, caption, greenlight_span_for_masked_lm_objective = self._compose_caption(target, box_labels, relation_labels_dict)
        tokens_positive = [obj["tokens_positive"] for obj in annotations]

        target.add_field("caption", caption)
        target.add_field("tokens_positive", tokens_positive)
        tokenized = self.tokenizer(caption, return_tensors="pt", max_length=max_query_len, truncation=True)
        target.add_field("positive_map", create_positive_map(tokenized, tokens_positive, max_len=max_query_len))
        target.add_field("positive_map_for_od_labels", create_positive_map_for_od_labels(tokenized, {}, max_len=max_query_len))
        target.add_field('greenlight_map', create_greenlight_map(greenlight_span_for_masked_lm_objective, tokenized, max_len=max_query_len))
        return target

    def _compose_caption(self, target, box_labels, relation_labels_dict):
        pseudo_caption = ""
        instance_to_positions = {}

        # relations
        relations = target.get_field("relation")
        for (s, o) in relations.nonzero():
            p_name = relation_labels_dict[(s.item(), o.item())]
            subj_name, obj_name = box_labels[s], box_labels[o]
            triplet_text = f"{subj_name} {p_name} {obj_name}. "
            pseudo_caption += triplet_text

            subj_pos = [len(pseudo_caption)-len(triplet_text), len(pseudo_caption)-len(triplet_text)+len(subj_name)]
            obj_pos = [len(pseudo_caption)-len(obj_name)-2, len(pseudo_caption)-2]
            for inst_id, inst_pos in zip([s.item(), o.item()], [subj_pos, obj_pos]):
                if inst_id in instance_to_positions:
                    instance_to_positions[inst_id].append(inst_pos)
                else:
                    instance_to_positions[inst_id] = [inst_pos]

        # return info
        new_target = []
        greenlight_span_for_masked_lm_objective = []
        for inst_id in range(len(target)):
            # add other instances not involved in relations
            if inst_id not in instance_to_positions:
                obj_name = box_labels[inst_id]
                obj_text = f"{obj_name}. "
                pseudo_caption += obj_text
                instance_to_positions[inst_id] = [[len(pseudo_caption)-len(obj_name)-2, len(pseudo_caption)-2]]

            new_target_i = {}
            new_target_i["tokens_positive"] = instance_to_positions[inst_id]
            new_target.append(new_target_i)
            greenlight_span_for_masked_lm_objective += instance_to_positions[inst_id]

        return new_target, pseudo_caption, greenlight_span_for_masked_lm_objective