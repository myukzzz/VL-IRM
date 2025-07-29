# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, get_rank
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
from .utils import check_data

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False,
        train_ov_relation=False,
        test_ov_relation=False,
        use_CLIPtext=False

):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)

    use_train = cfg.SOLVER.usetrain

    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()


    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1



    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):###################
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        if iteration == 20:
            print("#############################################################")
            print("train_ov_relation:", train_ov_relation)
            print("#############################################################")
        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]  #####四个都一样，全是object的类别
            captions_rel = [t.get_field("caption_rel") for t in targets if "caption_rel" in t.fields()]
            captions_rel_all = [t.get_field("all_rels") for t in targets if "all_rels" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:########################
                model.language_backbone.eval()

        # # check data
        # check_data(data_loader.dataset, images.tensors[0], targets[0])
        if use_train:
            if cfg.SOLVER.USE_AMP:
                with autocast():
                    if len(captions) > 0:
                        loss_dict,train_results = model(images, targets, captions, positive_map, greenlight_map = greenlight_map,captions_rel=captions_rel,train_ov_relation=train_ov_relation,use_CLIPtext=use_CLIPtext,captions_rel_all=captions_rel_all,now_iter=iteration)#GeneralizedVLRCNN
                    else:
                        loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                #############visualization##################################
                #########test##########
                # output_result = [o.to(cpu_device) for o in output]
                ####################
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
                # #rel_id = output_result[0].extra_fields['pred_rel_cls_topk']
                # extra_description={}########self.num_beams = 5 description比较多  text_len 3
                # all_rel_text = captions_rel[0].split('. ')
                # if 'pseudo_text' in train_results:
                #     label_texts = train_results['pseudo_text']
                #     tgt_texts = train_results['tgt_text']
                # else:
                #     label_texts=[]
                #     tgt_texts=[]
                # if len(label_texts)>0:
                #     for texts in label_texts:
                #         if texts not in seen_rels:
                #             if texts not in extra_description:
                #                 extra_description[texts]=0
                #             else:
                #                 extra_description[texts]=extra_description[texts]+1
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
                #
                #
                # def print_list_txt(name, input_list, f):
                #     for i, item in enumerate(input_list):
                #         f.write(name + ' ' + str(i) + ': ' + str(item) + '\n')
                #
                #
                # all_box_text = captions[0].split('. ')
                # image_name = targets[0].extra_fields['filename']
                # obj_labels = ['{}-{}'.format(idx, all_box_text[i-1]) for idx, i in
                #           enumerate(targets[0].extra_fields['labels'].tolist())]
                #
                # labels =[]
                # pred_rels = []
                # pred_rels_tgt = []
                # if len(label_texts) > 0:
                #     for idx_psedo, rel_psedo in enumerate(label_texts):
                #         if rel_psedo not in seen_rels:
                #             labels.append('{}-{}-new category'.format(idx_psedo, rel_psedo))
                #         else:
                #             labels.append('{}-{}'.format(idx_psedo, rel_psedo))
                #     pred_rels = [(obj_labels[train_results['sampled_inds_x'][obj_idx]], label, obj_labels[train_results['sampled_inds_y'][obj_idx]]) for obj_idx, label in
                #                  zip(train_results['pseudo_ids'],labels)]
                #
                #
                # tgt_labels=[]
                # if len(tgt_texts) > 0:
                #     for idx_tgt, rel_tgt in enumerate(tgt_texts):
                #         if rel_tgt not in seen_rels:
                #             tgt_labels.append('{}-{}-new category'.format(idx_tgt, rel_tgt))
                #         else:
                #             tgt_labels.append('{}-{}'.format(idx_tgt, rel_tgt))
                #     pred_rels_tgt = [(obj_labels[train_results['sampled_inds_x'][obj_idx]], label, obj_labels[train_results['sampled_inds_y'][obj_idx]]) for obj_idx, label in
                #                  zip(train_results['tgt_ids'],tgt_labels)]
                #
                #
                # boxes=targets[0].bbox
                # from PIL import Image, ImageDraw
                # pic = Image.open(image_name)
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
                # copy_image = pic.copy()
                # ################save#################
                #
                # copy_image.save(f'/data/myk/myk/openSGG/VS3/visualization/train/{iteration}.jpg')
                # if len(tgt_texts) > 0:
                #     f = open(f'/data/myk/myk/openSGG/VS3/visualization/train/output_pseudo_{iteration}.txt', 'w')
                #     print_list_txt('pred_rels', pred_rels[:100], f)
                #     f.close()
                #     f1 = open(f'/data/myk/myk/openSGG/VS3/visualization/train/output_tgt_{iteration}.txt', 'w')
                #     print_list_txt('pred_rels', pred_rels_tgt[:100], f1)
                #     f1.close()


                #############################################################







                # save checkpoints for further debug if nan happens
                # loss_value = losses.item()
                # if not math.isfinite(loss_value):
                #     logging.error(f'=> loss is {loss_value}, stopping training')
                #     logging.error("Losses are : {}".format(loss_dict))
                #     time_str = time.strftime('%Y-%m-%d-%H-%M')
                #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
                #     logging.info(f'=> save error state to {fname}')
                #     dict_to_save = {
                #         'x': images,
                #         'y': targets,
                #         'loss': losses,
                #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                #     }
                #     if len(captions) > 0:
                #         dict_to_save['captions'] = captions
                #         dict_to_save['positive_map'] = positive_map
                #     torch.save(
                #             dict_to_save,
                #             fname
                #         )


                if torch.isnan(losses) or torch.isinf(losses):
                    logging.error("NaN encountered, ignoring")
                    losses[losses != losses] = 0
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                if len(captions) > 0:
                    loss_dict = model(images, targets, captions, positive_map)
                else:
                    loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # loss_value = losses.item()
                # if not math.isfinite(loss_value):
                #     logging.error(f'=> loss is {loss_value}, stopping training')
                #     time_str = time.strftime('%Y-%m-%d-%H-%M')
                #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
                #     logging.info(f'=> save error state to {fname}')
                #     dict_to_save = {
                #         'x': images,
                #         'y': targets,
                #         'loss': losses,
                #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                #     }
                #     if len(captions) > 0:
                #         dict_to_save['captions'] = captions
                #         dict_to_save['positive_map'] = positive_map
                #     torch.save(
                #         dict_to_save,
                #         fname
                #     )


                if torch.isnan(losses) or torch.isinf(losses):
                    losses[losses != losses] = 0
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                scheduler.step()

            # Adapt the weight decay: only support multiStepLR
            if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
                if milestone_target < len(scheduler.milestones):
                    next_milestone = list(scheduler.milestones)[milestone_target]
                else:
                    next_milestone = float('inf')
                if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                    gamma = scheduler.gamma
                    logger.info("Drop the weight decay by {}!".format(gamma))
                    for param in optimizer.param_groups:
                        if 'weight_decay' in param:
                            param['weight_decay'] *= gamma
                    # move the target forward
                    milestone_target += 1

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            if model_ema is not None:
                model_ema.update(model)
                arguments["model_ema"] = model_ema.state_dict()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
            # if iteration % 1 == 0 or iteration == max_iter:
                #logger.info(
                if global_rank <= 0:
                    print(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "wd: {wd:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer.param_groups[0]["lr"],
                            wd=optimizer.param_groups[0]["weight_decay"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )



            #########eval#####################
            if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
                if is_main_process():
                    print("Evaluating")
                eval_result = 0.0
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                mkdir(output_folder)

                model.eval()
                if cfg.SOLVER.TEST_WITH_INFERENCE:#############################
                    with torch.no_grad():
                        try:
                            _model = model.module#####多gpu
                        except:
                            _model = model
                        _result = inference(
                            model = _model,
                            data_loader = val_data_loader,
                            dataset_name=cfg.DATASETS.TEST[0],
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=output_folder,
                            cfg=cfg,
                            train_ov_relation=train_ov_relation,
                            base_caption=captions_rel[0]
                        )
                        if is_main_process():
                            if isinstance(_result, (list, tuple)):
                                eval_result = _result[0].results['bbox']['AP']
                            else:
                                eval_result = _result
                else:
                    results_dict = {}
                    cpu_device = torch.device("cpu")
                    for i, batch in enumerate(val_data_loader):
                        images, targets, image_ids, positive_map, *_ = batch
                        with torch.no_grad():
                            images = images.to(device)
                            if positive_map is None:
                                output = model(images)
                            else:
                                captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                                output = model(images, captions, positive_map)
                            output = [o.to(cpu_device) for o in output]
                        results_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output)}
                        )
                    all_predictions = all_gather(results_dict)
                    if is_main_process():
                        predictions = {}
                        for p in all_predictions:
                            predictions.update(p)
                        predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                        eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                                box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                        if cfg.DATASETS.CLASS_AGNOSTIC:
                            eval_result = eval_result.results['box_proposal']['AR@100']
                        else:
                            eval_result = eval_result.results['bbox']['AP']
                model.train()

                if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:#false
                    model_ema.ema.eval()
                    results_dict = {}
                    cpu_device = torch.device("cpu")
                    for i, batch in enumerate(val_data_loader):
                        images, targets, image_ids, positive_map, positive_map_eval = batch
                        with torch.no_grad():
                            images = images.to(device)
                            if positive_map is None:
                                output = model_ema.ema(images)
                            else:
                                captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                                output = model_ema.ema(images, captions, positive_map)
                            output = [o.to(cpu_device) for o in output]
                        results_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output)}
                        )
                    all_predictions = all_gather(results_dict)
                    if is_main_process():
                        predictions = {}
                        for p in all_predictions:
                            predictions.update(p)
                        predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                        eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                                  box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                        if cfg.DATASETS.CLASS_AGNOSTIC:
                            eval_result = eval_result.results['box_proposal']['AR@100']
                        else:
                            eval_result = eval_result.results['bbox']['AP']

                arguments.update(eval_result=eval_result)

                if cfg.SOLVER.USE_AUTOSTEP:#false
                    eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                    # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                    scheduler.step(eval_result)

                if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:#false
                    if eval_result < previous_best:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                        previous_best = eval_result
                        checkpointer.save("model_best", **arguments)
                    print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                    if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                        if is_main_process():
                            print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                        break

            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
                break
        else:
            if val_data_loader:
                if is_main_process():
                    print("Evaluating")
                eval_result = 0.0
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                mkdir(output_folder)

                model.eval()
                if cfg.SOLVER.TEST_WITH_INFERENCE:  #############################
                    with torch.no_grad():
                        try:
                            _model = model.module  #####多gpu
                        except:
                            _model = model

                        _result = inference(
                            model=_model,
                            data_loader=val_data_loader,
                            dataset_name=cfg.DATASETS.TEST[0],
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=output_folder,
                            cfg=cfg,
                            train_ov_relation=train_ov_relation,
                            base_caption=captions_rel[0]
                        )


                        ####val on train
                        # _result = inference(
                        #     model=_model,
                        #     data_loader=data_loader,
                        #     dataset_name=cfg.DATASETS.TEST[0],
                        #     device=cfg.MODEL.DEVICE,
                        #     expected_results=cfg.TEST.EXPECTED_RESULTS,
                        #     expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        #     output_folder=output_folder,
                        #     cfg=cfg,
                        #     train_ov_relation=train_ov_relation,
                        #     base_caption=captions_rel[0]
                        # )

                        if is_main_process():
                            if isinstance(_result, (list, tuple)):
                                eval_result = _result[0].results['bbox']['AP']
                            else:
                                eval_result = _result
                else:
                    results_dict = {}
                    cpu_device = torch.device("cpu")
                    for i, batch in enumerate(val_data_loader):
                        images, targets, image_ids, positive_map, *_ = batch
                        with torch.no_grad():
                            images = images.to(device)
                            if positive_map is None:
                                output = model(images)
                            else:
                                captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                                output = model(images, captions, positive_map)
                            output = [o.to(cpu_device) for o in output]
                        results_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output)}
                        )
                    all_predictions = all_gather(results_dict)
                    if is_main_process():
                        predictions = {}
                        for p in all_predictions:
                            predictions.update(p)
                        predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                        eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                                  box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                        if cfg.DATASETS.CLASS_AGNOSTIC:
                            eval_result = eval_result.results['box_proposal']['AR@100']
                        else:
                            eval_result = eval_result.results['bbox']['AP']
                model.train()

                if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:  # false
                    model_ema.ema.eval()
                    results_dict = {}
                    cpu_device = torch.device("cpu")
                    for i, batch in enumerate(val_data_loader):
                        images, targets, image_ids, positive_map, positive_map_eval = batch
                        with torch.no_grad():
                            images = images.to(device)
                            if positive_map is None:
                                output = model_ema.ema(images)
                            else:
                                captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                                output = model_ema.ema(images, captions, positive_map)
                            output = [o.to(cpu_device) for o in output]
                        results_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output)}
                        )
                    all_predictions = all_gather(results_dict)
                    if is_main_process():
                        predictions = {}
                        for p in all_predictions:
                            predictions.update(p)
                        predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                        eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                                  box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                        if cfg.DATASETS.CLASS_AGNOSTIC:
                            eval_result = eval_result.results['box_proposal']['AR@100']
                        else:
                            eval_result = eval_result.results['bbox']['AP']

                arguments.update(eval_result=eval_result)

                if cfg.SOLVER.USE_AUTOSTEP:  # false
                    eval_result = all_gather(eval_result)[0]  # broadcast_data([eval_result])[0]
                    # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                    scheduler.step(eval_result)

                if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:  # false
                    if eval_result < previous_best:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                        previous_best = eval_result
                        checkpointer.save("model_best", **arguments)
                    print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result",
                          eval_result)
                    if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                        if is_main_process():
                            print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration,
                                                                                                 previous_best))
                        break
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
