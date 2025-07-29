# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import re

import torch
from tqdm import tqdm
from collections import defaultdict
from maskrcnn_benchmark.data.datasets.evaluation import evaluate, im_detect_bbox_aug
from ..utils.comm import is_main_process, get_rank, all_gather, synchronize, get_world_size
from maskrcnn_benchmark.data.datasets.evaluation.flickr.flickr_eval import FlickrEvaluator
import matplotlib.pyplot as plt
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
from .utils import check_data

def inference_default(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, *_ = batch
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    predictions = results_dict
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return None

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def create_one_hot_dict(labels, no_minus_one_for_one_hot = False):
    positive_map_token_to_label = defaultdict(int)
    positive_map_label_to_token = defaultdict(int)

    for i in range(len(labels)):
        positive_map_token_to_label[i] = labels[i]
        positive_map_label_to_token[labels[i]] = i

    if no_minus_one_for_one_hot:
        positive_map_token_to_label = defaultdict(int)
        positive_map_label_to_token = defaultdict(int)

        for i in range(len(labels)):
            positive_map_token_to_label[i+1] = labels[i]
            positive_map_label_to_token[labels[i]] = i + 1

    return positive_map_token_to_label, positive_map_label_to_token


def create_positive_dict(tokenized, tokens_positive, labels,tokenized_rel,tokens_rel,rel_labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)
    positive_map_rel = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = defaultdict(list)
    positive_map_label_to_token_rel = defaultdict(list)


    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)####label的token位置
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)

    for j, tok_list in enumerate(tokens_rel):
        for (beg, end) in tok_list:
            beg_pos = tokenized_rel.char_to_token(beg)
            end_pos = tokenized_rel.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized_rel.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized_rel.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized_rel.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized_rel.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map_rel[i] = rel_labels[j]  # because the labels starts from 1
                positive_map_label_to_token_rel[rel_labels[j]].append(i)####label的token位置
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map, positive_map_label_to_token,positive_map_rel,positive_map_label_to_token_rel  # / (positive_map.sum(-1)[:, None] + 1e-6)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

def create_queries_and_maps_from_dataset(dataset, cfg):
    categories = dataset.categories()
    categories_rel = dataset.ind_to_predicates
    #one_hot = dataset.one_hot

    labels = []
    label_list = []
    keys = list(categories.keys())
    keys.sort()
    rel_label_list=categories_rel[1:]
    rel_labels=[]

    for i in range(len(rel_label_list)):
        rel_labels.append(i+1)

    for i in keys:
        labels.append(i)
        label_list.append(categories[i])

    if cfg.TEST.CHUNKED_EVALUATION != -1:
        labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
        label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
    else:###########
        labels = [labels]
        label_list = [label_list]

    all_queries = []
    all_positive_map_label_to_token = []

    all_queries_rel = []
    all_positive_map_label_to_token_rel = []

    for i in range(len(labels)):
        labels_i = labels[i]
        label_list_i = label_list[i]
        query_i, positive_map_label_to_token_i,query_i_rel, positive_map_label_to_token_i_rel = create_queries_and_maps(
            labels_i, label_list_i,rel_label_list=rel_label_list,rel_labels=rel_labels, additional_labels = cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg = cfg)
        
        all_queries.append(query_i)
        all_positive_map_label_to_token.append(positive_map_label_to_token_i)

        all_queries_rel.append(query_i_rel)
        all_positive_map_label_to_token_rel.append(positive_map_label_to_token_i_rel)
    print("All queries", all_queries)
    return all_queries, all_positive_map_label_to_token,all_queries_rel, all_positive_map_label_to_token_rel

def create_queries_and_maps(labels, label_list, additional_labels = None, cfg = None,rel_labels=None,rel_label_list=None):

    # Clean label list
    original_label_list = label_list.copy()
    label_list = [clean_name(i) for i in label_list]#150
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""
    rel_query = ""
    tokens_rel=[]

    # sep between tokens, follow training
    separation_tokens = cfg.DATASETS.SEPARATION_TOKENS
    
    caption_prompt = cfg.DATASETS.CAPTION_PROMPT
    if caption_prompt is not None and isinstance(caption_prompt, str):
        caption_prompt = load_from_yaml_file(caption_prompt)
    use_caption_prompt = cfg.DATASETS.USE_CAPTION_PROMPT and caption_prompt is not None#false
    for _index, label in enumerate(label_list):
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["prefix"]
        
        start_i = len(objects_query)

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["name"]
        else:
            objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]##object的token坐标
        
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["suffix"]

        if _index != len(label_list) - 1:#用.分隔
            objects_query += separation_tokens

    for idx, relabel in enumerate(rel_label_list):
        start_index = len(rel_query)
        rel_query += relabel
        end_index = len(rel_query)

        tokens_rel.append([(start_index, end_index)])#######记录label的str的位置

        if idx != len(rel_label_list) - 1:
            rel_query += separation_tokens



    if additional_labels is not None:
        objects_query += separation_tokens
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += separation_tokens

    print(objects_query)

    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":##############
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_LOCAL_FILES_ONLY)
        tokenized = tokenizer(objects_query, return_tensors="pt")#308

        tokenized_rel = tokenizer(rel_query, return_tensors="pt")#308

    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        from transformers import CLIPTokenizerFast
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True, mask_token='ðŁĴĳ</w>')
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True)
        tokenized = tokenizer(objects_query,
                              max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                              truncation=True,
                              return_tensors="pt")
    else:
        tokenizer = None
        raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token,positive_map_token_to_label_rel, positive_map_label_to_token_rel = create_positive_dict(tokenized, tokens_positive,
                                                                                        labels=labels,tokenized_rel=tokenized_rel,tokens_rel=tokens_rel,rel_labels=rel_labels)  # from token position to original label
    return objects_query, positive_map_label_to_token,rel_query,positive_map_label_to_token_rel

def create_positive_map_label_to_token_from_positive_map(positive_map, plus = 0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token



def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def resize_box(output, targets):
    if isinstance(targets[0], dict):
        orig_target_sizes = targets[0]["orig_size"].unsqueeze(0)
    else:
        orig_target_sizes = torch.stack([targets[0].extra_fields["orig_size"] for _ in range(1)], dim=0)
    img_h, img_w = orig_target_sizes.unbind(1)
    return output.resize((img_w, img_h))

def flickr_post_process(output, targets, positive_map_label_to_token, plus):
    output = resize_box(output, targets)
    scores, indices = torch.topk(output.extra_fields["scores"], k = len(output.extra_fields["scores"]), sorted=True)
    boxes = output.bbox.tolist()
    boxes = [boxes[i] for i in indices]
    labels = [output.extra_fields["labels"][i] for i in indices]
    output_boxes = [[] for i in range(len(positive_map_label_to_token))]
    output_scores = [[] for i in range(len(positive_map_label_to_token))]
    for i in range(len(boxes)):
        output_boxes[labels[i] - plus].append(boxes[i])
        output_scores[labels[i] - plus].append(scores[i])
    for i in output_boxes:
        i.append([0.0, 0.0, 0.0, 0.0])
    image_ids = [t.extra_fields["original_img_id"] for t in targets]
    sentence_ids = [t.extra_fields["sentence_id"] for t in targets]

    return {"image_id": image_ids[0], "sentence_id": sentence_ids[0], "boxes": output_boxes, "scores": output_scores}

def build_flickr_evaluator(cfg):
    evaluator = FlickrEvaluator(
        "DATASET/flickr30k/flickr30k_entities/", # Hard written!!
        subset="test" if "test" in cfg.DATASETS.TEST[0]  else "val",
        merge_boxes=cfg.DATASETS.FLICKR_GT_TYPE == "merged")
    return evaluator

def build_lvis_evaluator(ann_file, fixed_ap=True):
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis import LVIS
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis_eval import LvisEvaluatorFixedAP, LvisEvaluator
    evaluator = LvisEvaluatorFixedAP(LVIS(ann_file), fixed_ap=fixed_ap)
    #evaluator = LvisEvaluator(LVIS(ann_file), iou_types=['segm', 'bbox'])
    return evaluator

def write_lvis_results(results, output_file_name):
    lines = []
    lines.append("metric, avg ")
    for each_result in results:
        metric_string = " ".join(each_result.split(" ")[:-2])
        number = each_result.split(" ")[-1]
        each_result = metric_string + ", " + number + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return

def write_flickr_results(results, output_file_name):
    '''
    {'Recall@1_all': 0.8394651146677753, 'Recall@1_animals': 0.9177820267686424, 'Recall@1_bodyparts': 0.7097966728280961, ...}
    '''
    lines = []
    lines.append("metric, avg ")
    for each_metric, number in results.items():
        each_result = each_metric + ", " + str(number) + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return




def show_object(obj_prediction=None):
    aaaa=0




    return








def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None,
        verbose=True,
        visualizer = None,
        train_ov_relation=False,
        base_caption=None
):
    # convert to a torch.device for efficiency
    try:
        device = torch.device(device)
    except:
        device = device
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    if verbose:##################
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    task = cfg.TEST.EVAL_TASK

    if not task:
        return inference_default(model, data_loader, dataset_name, iou_types, box_only, device, expected_results, expected_results_sigma_tol, output_folder, cfg)
        
    if cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
        assert task == 'detection'
        categories = dataset.categories()

        keys = list(categories.keys())
        keys.sort()
        all_queries = [[categories[k] for k in keys]]
        all_positive_map_label_to_token = [{k: [i] for i, k in enumerate(keys)}]
    elif task == "detection":#################
        all_queries, all_positive_map_label_to_token,all_queries_rel, all_positive_map_label_to_token_rel = create_queries_and_maps_from_dataset(dataset, cfg)
    elif task == "grounding":
        all_queries = [None]
        all_positive_map_label_to_token = [None]
    else:
        assert(0)

    '''
    Build Dataset Sepecific Evaluator
    '''
    if "flickr" in cfg.DATASETS.TEST[0]:
        evaluator = build_flickr_evaluator(cfg)
    elif "lvis" in cfg.DATASETS.TEST[0]:
        evaluator = build_lvis_evaluator(dataset.ann_file, fixed_ap=not cfg.DATASETS.LVIS_USE_NORMAL_AP)
    else:
        evaluator = None

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    if verbose:#############
        _iterator = tqdm(data_loader)
    else:
        _iterator = data_loader
    #
    extra_description = {}
    def write_dict_to_txt(data_dict, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for key, value in data_dict.items():
                f.write(f"{key}: {value}\n")
    for i, batch in enumerate(_iterator):

        #####myeval####
        # if i <510:
        #     continue


        if i == cfg.TEST.SUBSET:
            break
        images, targets, image_ids, *_ = batch
        if cfg.TEST.myeval:
            if i == 10:
                break
        all_output = []
        mdetr_style_output = []
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                query_time = len(all_queries)
                for query_i in range(query_time):
                    if task == "detection":##################
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                    else:
                        captions = None
                        positive_map_label_to_token = None

                output = im_detect_bbox_aug(model, images, device, captions, positive_map_label_to_token)
                output = [o.to(cpu_device) for o in output]
                all_output.append(output)
            else:
                images = images.to(device)
                query_time = len(all_queries)

                for query_i in range(query_time):
                    if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
                        targets = [target.to(device) for target in targets]
                    '''
                    different datasets seem to have different data format... For LVIS dataset, the target is a dictionary, while for modulatedDataset such as COCO/Flickr, the target is a BoxList
                    '''

                    if task == "detection":#####################
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]

                        captions_rel = [all_queries_rel[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token_rel = all_positive_map_label_to_token_rel[query_i]

                    elif task == "grounding":
                        captions = [t.get_field("caption") for t in targets]
                        positive_map_eval = [t.get_field("positive_map_eval") for t in targets]
                        if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                            plus = 1
                        else:
                            plus = 0
                        assert(len(positive_map_eval) == 1) # Let's just use one image per batch
                        positive_map_eval = positive_map_eval[0]
                        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map_eval, plus=plus)
                    output = model(images, captions=captions, positive_map=positive_map_label_to_token, targets=targets,captions_rel=captions_rel,positive_map_label_to_token_rel=positive_map_label_to_token_rel,train_ov_relation=train_ov_relation,use_CLIPtext=cfg.SOLVER.use_CLIPtext)####caption改为150




                    ####################################visualization#####################################
                    # output_result = [o.to(cpu_device) for o in output]
                    # seen_rels = []
                    # torch.manual_seed(0)
                    # rand_relation_idx = torch.randperm(51 - 1)  # 44 39 7 6 17 29
                    # base_rel_idx_ori = rand_relation_idx[:int((51 - 1) * 0.5)].tolist()
                    #
                    # VG150_REL_CATEGORIES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at',
                    #                         'attached to', 'behind', 'belonging to', 'between', 'carrying',
                    #                         'covered in', 'covering', 'eating', 'flying in', 'for', 'from',
                    #                         'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of',
                    #                         'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near',
                    #                         'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of',
                    #                         'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under',
                    #                         'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
                    # base_rel_idx = [l + 1 for l in base_rel_idx_ori]
                    #
                    # unseen_rels = [9, 36, 38, 45, 22, 30, 34, 37, 16, 46, 39, 7, 14, 31, 49]
                    # unseen_rels_onlyrel = [unseen_rel - 1 for unseen_rel in unseen_rels]
                    #
                    # base_rels_new = []
                    # base_rels_new_idx = list(set(rand_relation_idx.tolist()) - set(unseen_rels_onlyrel))
                    #
                    # base_rels_new.append(VG150_REL_CATEGORIES[0])
                    # for base_rels in base_rels_new_idx:
                    #     base_rels_new.append(VG150_REL_CATEGORIES[base_rels + 1])
                    # for k in base_rel_idx:
                    #     seen_rels.append(VG150_REL_CATEGORIES[k])
                    #
                    #
                    #
                    # rel_id = output_result[0].extra_fields['pred_rel_cls_topk']
                    # ########self.num_beams = 5 description比较多  text_len 3
                    # all_rel_text = captions_rel[0].split('. ')
                    # label_texts=[all_rel_text[relid-1]for relid_idx,relid in enumerate(rel_id)]
                    #
                    #
                    # def draw_single_box(pic, box, color='red', draw_info=None):
                    #     color_set=['tomato','gold','pink','deepskyblue','lightgreen']
                    #     import random
                    #     color = random.sample(color_set, 1)
                    #     draw = ImageDraw.Draw(pic)
                    #     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    #     draw.rectangle(((x1, y1), (x2, y2)), outline=color[0],width=1)
                    #
                    #     #draw.rectangle(((x1, y1), (x2, y2)), pic, (0, 155, 875, 435), fill = (255, 255, 255, 150))
                    #
                    #
                    #
                    #
                    #
                    #     #draw.rectangle(((x1, y1), (x2, y2)), fill=(200, 100, 0, 127))
                    #     #draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 0, 127), width=3)
                    #     if draw_info:
                    #         draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color[0])
                    #         info = draw_info
                    #         draw.text((x1, y1), info)
                    # def print_list_txt(name, input_list, f):
                    #     for i, item in enumerate(input_list):
                    #         f.write(name + ' ' + str(i) + ': ' + str(item) + '\n')
                    #
                    #
                    #
                    # #pair_ids=output[0].extra_fields['rel_pair_idxs_topk']
                    # #object_labels=output[0].extra_fields['labels']
                    #
                    # all_box_text = captions[0].split('. ')
                    image_name = targets[0].extra_fields['filename']
                    # obj_labels = ['{}-{}'.format(idx, all_box_text[i-1]) for idx, i in
                    #           enumerate(output_result[0].extra_fields['labels'].tolist())]
                    #
                    # labels =[]
                    # generate_labels=[]
                    # for top_beam in range(len(output[0].extra_fields['decoder_output_topk'][0])):
                    #     generate_labels.append([])
                    # for idx, rel in enumerate(rel_id.tolist()):
                    #     if all_rel_text[rel-1] not in base_rels_new:
                    #         labels.append('{}-{}-new category'.format(idx, all_rel_text[rel-1]))
                    #
                    #
                    #     else:
                    #         labels.append('{}-{}'.format(idx, all_rel_text[rel - 1]))
                    #     # labels = ['{}-{}'.format(idx, all_rel_text[i-1]) for idx, i in
                    #     #       enumerate(rel_id.tolist())]
                    #
                    #
                    # new_count=0
                    # for generate_idx,generate_label in enumerate(output[0].extra_fields['decoder_output_topk']):
                    #     for label_top in range(len(generate_label)):
                    #         if generate_label[label_top] not in VG150_REL_CATEGORIES:
                    #             new_count=new_count+1
                    #             generate_labels[label_top].append('{}-{}-new category'.format(generate_idx, generate_label[label_top]))
                    #             if generate_label[label_top] not in extra_description:
                    #                 extra_description[generate_label[label_top]] = 0
                    #             else:
                    #                 extra_description[generate_label[label_top]] = extra_description[generate_label[label_top]] + 1
                    #
                    #         else:
                    #             generate_labels[label_top].append('{}-{}'.format(generate_idx, generate_label[label_top]))
                    #
                    # pred_rels = [(obj_labels[obj_idx[0]], label, obj_labels[obj_idx[1]]) for obj_idx, label in
                    #              zip(output[0].extra_fields['rel_pair_idxs_topk'].tolist(),labels)]
                    # boxes=output_result[0].bbox
                    from PIL import Image, ImageDraw
                    pic = Image.open(image_name)
                    # num_obj = boxes.shape[0]
                    # for j in range(num_obj):
                    #     info = obj_labels[j]
                    #     pic_height,pic_len=pic.size[0],pic.size[1]
                    #     draw_height,draw_len=targets[0].size[0],targets[0].size[1]
                    #     height_scale,len_scale=pic_height/draw_height,pic_len/draw_len
                    #     boxes[j][0] = boxes[j][0] * height_scale
                    #     boxes[j][1] = boxes[j][1] * len_scale
                    #     boxes[j][2] = boxes[j][2] * height_scale
                    #     boxes[j][3] = boxes[j][3] * len_scale
                    #
                    #     draw_single_box(pic, boxes[j], draw_info=info)
                    copy_image = pic.copy()
                    ################save#################

                    copy_image.save(f'/data/myk/myk/openSGG/VS3/visualization/testori/{i}.jpg')
                    # f = open(f'/data/myk/myk/openSGG/VS3/visualization/test_10/output{i}.txt', 'w')
                    # print_list_txt('pred_rels', pred_rels[:100], f)
                    # f.close()
                    # if new_count>0:
                    #     f1 = open(f'/data/myk/myk/openSGG/VS3/visualization/test_10/output{i}_generate.txt', 'w')
                    #
                    #     for top_idx in range(len(output[0].extra_fields['decoder_output_topk'][0])):
                    #         pred_rels_gen = [(obj_labels[obj_idx[0]], label, obj_labels[obj_idx[1]]) for obj_idx, label in
                    #                      zip(output[0].extra_fields['rel_pair_idxs_topk'].tolist(), generate_labels[top_idx])]
                    #         print_list_txt('pred_rels', pred_rels_gen, f1)
                    #     f1.close()
                    #
                    # write_dict_to_txt(extra_description, '/data/myk/myk/openSGG/VS3/visualization/all_new_top10.txt')
                    ######################################################################################################################################################


                    if "flickr" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        new_output = flickr_post_process(
                            output,
                            targets,
                            positive_map_label_to_token,
                            plus # This is only used in Flickr
                        )
                        mdetr_style_output.append(new_output)
                    elif "lvis" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        output = resize_box(output, targets)
                        scores = output.extra_fields["scores"]
                        labels = output.extra_fields["labels"]
                        boxes = output.bbox
                        mdetr_style_output.append((targets[0]["image_id"].item(), {"scores": scores, "labels": labels, "boxes": boxes}))
                    else:
                        all_output.append(output)

        # # check target and prediction
        # check_data(data_loader.dataset, images.tensors[0], targets[0])
        # check_data(data_loader.dataset, images.tensors[0], output[0], mode='prediction', rel_num=20)

        if evaluator is not None:
            evaluator.update(mdetr_style_output)
        else:
            output = [[row[_i] for row in all_output] for _i in range(len(all_output[0]))]
            for index, i in enumerate(output):
                output[index][0].extra_fields.pop('decoder_output_topk', None)
                output[index] = i[0].concate_box_list(i)
                output[index].extra_fields.pop('box_features', None)

            results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})




    if evaluator is not None:
        evaluator.synchronize_between_processes()
        try:
            evaluator.accumulate()
        except:
            print("Evaluator has no accumulation, skipped...")
        score = evaluator.summarize()
        print(score)
        import maskrcnn_benchmark.utils.mdetr_dist as dist
        if is_main_process():
            if "flickr" in cfg.DATASETS.TEST[0]:
                write_flickr_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
            elif "lvis" in cfg.DATASETS.TEST[0]:
                write_lvis_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
        try:
            torch.distributed.barrier()
        except:
            print("Default process group is not initialized")
        return

    if evaluator is not None:
        predictions = mdetr_style_output
    else:
        predictions = results_dict

    # wait for all processes to complete before measuring the time
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, f"predictions_rank={get_rank()}.pth"))
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    # print("Accumulating results")
    # merged_predictions = _accumulate_predictions_from_multiple_gpus(predictions) # may encounter GPU OOM
    if not is_main_process():
        return None

    # merge predictions in the main process (refer to _accumulate_predictions_from_multiple_gpus)
    merged_predictions = {}
    for rank in range(get_world_size()):
        rank_prediction = torch.load(os.path.join(output_folder, f"predictions_rank={rank}.pth"))
        merged_predictions.update(rank_prediction)

    image_ids = list(sorted(merged_predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    merged_predictions = [merged_predictions[i] for i in image_ids]

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    show_object(obj_prediction=merged_predictions)

    return evaluate(dataset=dataset, predictions=merged_predictions, output_folder=output_folder, config=cfg, **extra_args)
