import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from itertools import product
import sys
import torch.nn.functional as F
from itertools import combinations

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, feats, anchors, annotations, geos, batch_map):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        geo_on = True

        datas_graph = []
        
        for key, value in batch_map.items():
            unique_ids_ls = []
            dict_edge_gen = {}
            for j in range(value):
                classification = classifications[j, :, :]
                regression = regressions[j, :, :]
                features = feats[j, :, :]
                if geo_on == True:
                    geos = geos[j].repeat(1,features.shape[0]).view(-1,3)
                    features = torch.cat((features, geos.float()), 1)

                bbox_annotation = annotations[j, :, :]
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

                if bbox_annotation.shape[0] == 0:
                    if torch.cuda.is_available():
                        regression_losses.append(torch.tensor(0).float().cuda())
                        classification_losses.append(torch.tensor(0).float().cuda())
                    else:
                        regression_losses.append(torch.tensor(0).float())
                        classification_losses.append(torch.tensor(0).float())

                    continue

                classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

                # compute the loss for classification
                targets = torch.ones(classification.shape) * -1

                if torch.cuda.is_available():
                    targets = targets.cuda()

                targets[torch.lt(IoU_max, 0.4), :] = 0
                
                negative_indices = torch.lt(IoU_max, 0.15)
                positive_indices = torch.ge(IoU_max, 0.5)
                high_positive_indices = torch.ge(IoU_max, 0.7)

                num_positive_anchors = positive_indices.sum()

                assigned_annotations = bbox_annotation[IoU_argmax, :]

                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

                if torch.cuda.is_available():
                    alpha_factor = torch.ones(targets.shape).cuda() * alpha
                else:
                    alpha_factor = torch.ones(targets.shape) * alpha

                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

                # cls_loss = focal_weight * torch.pow(bce, gamma)
                cls_loss = focal_weight * bce

                if torch.cuda.is_available():
                    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
                else:
                    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

                classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

                # compute the loss for regression

                if positive_indices.sum() > 0:
                    assigned_annotations_ = assigned_annotations[positive_indices, :]
                    # get instance ids and append to list
                    instance_ids = list(assigned_annotations[:,5].unique()[1:].cpu().numpy())
                    unique_ids_ls += instance_ids
                    unique_ids_ls = list(set(unique_ids_ls))

                    for inst_id in unique_ids_ls:
                        # boxes with instances matching inst_id
                        instance_boxes = torch.eq(assigned_annotations[:,5],inst_id)
                        # boxes with high IoU instances
                        if high_positive_indices.sum() == 0:
                            intersection = positive_indices * instance_boxes
                        else:
                            intersection = high_positive_indices * instance_boxes #this

                        if features[intersection].shape[0] == 0:
                            intersection = torch.ge(IoU_max, 0.5) * instance_boxes
                        
                        pos_feats = features[intersection]

                        pos_class_feats = classification[intersection]
                        pos_regression_feats = regression[intersection]

                        pos_geo_feats = assigned_annotations[intersection][:,8:]
                        neg_feats = features[~intersection]
                        neg_class_feats = classification[~intersection]
                        neg_regression_feats = regression[~intersection]
                        neg_geo_feats = assigned_annotations[~intersection][:,8:]

                        rand = torch.randperm(neg_feats.shape[0])[:pos_feats.shape[0]]

                        neg_feats = neg_feats[rand,:]
                        neg_class_feats = neg_class_feats[rand,:]
                        neg_regression_feats = neg_regression_feats[rand,:]
                        neg_geo_feats = neg_geo_feats[rand,:]

                        if pos_feats.shape[0] != 0:
                            if inst_id not in dict_edge_gen:
                                dict_edge_gen[inst_id] = {}

                            dict_edge_gen[inst_id][j] = {
                            "pos_feats": pos_feats,
                            "pos_class_feats": pos_class_feats,
                            "pos_regression_feats": pos_regression_feats,
                            "pos_geo_feats": pos_geo_feats,
                            "neg_feats": neg_feats,
                            "neg_class_feats": neg_class_feats,
                            "neg_regression_feats": neg_regression_feats,
                            "neg_geo_feats": neg_geo_feats}

                    anchor_widths_pi = anchor_widths[positive_indices]
                    anchor_heights_pi = anchor_heights[positive_indices]
                    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                    gt_widths = assigned_annotations_[:, 2] - assigned_annotations_[:, 0]
                    gt_heights = assigned_annotations_[:, 3] - assigned_annotations_[:, 1]
                    gt_ctr_x = assigned_annotations_[:, 0] + 0.5 * gt_widths
                    gt_ctr_y = assigned_annotations_[:, 1] + 0.5 * gt_heights

                    # clip widths to 1
                    gt_widths  = torch.clamp(gt_widths, min=1)
                    gt_heights = torch.clamp(gt_heights, min=1)

                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    targets_dh = torch.log(gt_heights / anchor_heights_pi)

                    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                    targets = targets.t()

                    if torch.cuda.is_available():
                        targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                    else:
                        targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                    negative_indices = 1 + (~positive_indices)

                    regression_diff = torch.abs(targets - regression[positive_indices, :])

                    regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / 9.0
                    )
                    regression_losses.append(regression_loss.mean())
                else:
                    if torch.cuda.is_available():
                        regression_losses.append(torch.tensor(0).float().cuda())
                    else:
                        regression_losses.append(torch.tensor(0).float())

            all_feats = torch.tensor([]).cuda()
            all_class_feats = torch.tensor([]).cuda()
            all_regression_feats = torch.tensor([]).cuda()
            all_geo_feats = torch.tensor([]).cuda()
            all_gt = torch.tensor([]).cuda()
            all_edges_index = torch.tensor([]).cuda().long()


            pos_dict = {}
            for inst_id, value in dict_edge_gen.items():
                #initialize temp tensors
                tmp_pos_feats = torch.tensor([]).cuda()
                tmp_pos_class_feats = torch.tensor([]).cuda()
                tmp_pos_regression_feats = torch.tensor([]).cuda()
                tmp_pos_geo_feats = torch.tensor([]).cuda()
                tmp_neg_feats = torch.tensor([]).cuda()
                tmp_neg_class_feats = torch.tensor([]).cuda()
                tmp_neg_regression_feats = torch.tensor([]).cuda()
                tmp_neg_geo_feats = torch.tensor([]).cuda()

                # for all batches inside instances
                for batch_id, value_inst in dict_edge_gen[inst_id].items():

                    tmp_pos_feats = torch.cat((tmp_pos_feats, dict_edge_gen[inst_id][batch_id]["pos_feats"]), 0)
                    tmp_pos_class_feats = torch.cat((tmp_pos_class_feats, dict_edge_gen[inst_id][batch_id]["pos_class_feats"]), 0)
                    tmp_pos_regression_feats = torch.cat((tmp_pos_regression_feats, dict_edge_gen[inst_id][batch_id]["pos_regression_feats"]), 0)
                    tmp_pos_geo_feats = torch.cat((tmp_pos_geo_feats, dict_edge_gen[inst_id][batch_id]["pos_geo_feats"]), 0)
                    tmp_neg_feats = torch.cat((tmp_neg_feats, dict_edge_gen[inst_id][batch_id]["neg_feats"]), 0)
                    tmp_neg_class_feats = torch.cat((tmp_neg_class_feats, dict_edge_gen[inst_id][batch_id]["neg_class_feats"]), 0)
                    tmp_neg_regression_feats = torch.cat((tmp_neg_regression_feats, dict_edge_gen[inst_id][batch_id]["neg_regression_feats"]), 0)
                    tmp_neg_geo_feats = torch.cat((tmp_neg_geo_feats, dict_edge_gen[inst_id][batch_id]["neg_geo_feats"]), 0)

                all_nodes_range_start = all_feats.shape[0]
                pos_nodes_range_start = tmp_pos_feats.shape[0]
                neg_nodes_range_start = tmp_neg_feats.shape[0]
                pos_node = torch.arange(all_nodes_range_start, all_nodes_range_start + pos_nodes_range_start)
                neg_node = torch.arange(all_nodes_range_start + pos_nodes_range_start, all_nodes_range_start + pos_nodes_range_start + neg_nodes_range_start)

                pos_dict[inst_id] = pos_node

                # create combination of positive nodes of all anchors inside the instance
                edges_pos_nat = torch.combinations(pos_node, 2)

                # create the opposite edge/relationship            
                edges_pos_rev = torch.stack((edges_pos_nat[:,1],edges_pos_nat[:,0]),1)
                edge_pos_ = torch.cat((edges_pos_nat,edges_pos_rev),0)

                # create gt for pos
                y_pos = torch.ones(edge_pos_.shape[0]).cuda()

                numpy_pos_node = pos_node.detach().cpu().numpy()
                numpy_neg_node = neg_node.detach().cpu().numpy()
                edges_neg_pos_combo = torch.tensor(np.array(list(product(numpy_pos_node,numpy_neg_node))))
                
                edges_neg_pos_combo_rev = torch.stack((edges_neg_pos_combo[:,1],edges_neg_pos_combo[:,0]),1)
                edge_neg_pos_ = torch.cat((edges_neg_pos_combo,edges_neg_pos_combo_rev),0)

                y_neg = torch.zeros(edge_neg_pos_.shape[0]).cuda()

                feats_pos_neg = torch.cat((tmp_pos_feats, tmp_neg_feats), 0)    
                class_feats = torch.cat((tmp_pos_class_feats, tmp_neg_class_feats), 0)
                regression_feats = torch.cat((tmp_pos_regression_feats, tmp_neg_regression_feats), 0)
                geo_feats = torch.cat((tmp_pos_geo_feats, tmp_neg_geo_feats), 0)
                inst_edges = torch.cat((edge_pos_, edge_neg_pos_), 0).cuda().long()

                y_ = torch.cat((y_pos, y_neg), 0).cuda()
                
                all_edges_index = torch.cat((all_edges_index, inst_edges), 0).cuda().long()
                all_feats = torch.cat((all_feats, feats_pos_neg), 0).cuda()
                all_geo_feats = torch.cat((all_geo_feats, geo_feats), 0).cuda()
                all_regression_feats = torch.cat((all_regression_feats, regression_feats), 0).cuda()
                all_gt = torch.cat((all_gt, y_), 0)

            pos_comb = combinations(pos_dict.items(), 2)
            for comb in list(pos_comb):
                pos1 = comb[0][1].detach().cpu().numpy()
                pos2 = comb[1][1].detach().cpu().numpy()
                edges_pos_pos_neg_combo = torch.tensor(np.array(list(product(pos1,pos2))))
                edges_pos_pos_neg_combo_rev = torch.stack((edges_pos_pos_neg_combo[:,1],edges_pos_pos_neg_combo[:,0]),1)
                edge_neg_neg_ = torch.cat((edges_pos_pos_neg_combo,edges_pos_pos_neg_combo_rev),0)
                y_neg = torch.zeros(edge_neg_neg_.shape[0]).cuda()
                all_edges_index = torch.cat((all_edges_index, edge_neg_neg_.cuda().long()), 0)
                all_gt = torch.cat((all_gt, y_neg), 0)

            try:
                # Debug me
                all_edges_index = all_edges_index.view(all_edges_index.shape[1], all_edges_index.shape[0])
            except IndexError as error:
                print("Index error HAPPENING with no reason")

            data_ = Data(classification = all_class_feats,
                regressions = all_regression_feats,
                geos= all_geo_feats,
                x = all_feats.float(),
                edge_index = all_edges_index,
                y = all_gt.cuda().double()
            )
            # print(data_)
            datas_graph.append(data_)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True), datas_graph
