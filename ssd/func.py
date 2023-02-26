from torch.autograd import Variable
import torch
from contextlib import redirect_stdout
import io
from pycocotools.cocoeval import COCOeval

def model_func(model, data, forward_info):
    if not forward_info['is_inference']:
        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        bbox_offsets = bbox_offsets.cuda()

        img.sub_(forward_info['mean']).div_(forward_info['std'])
        if not forward_info['no_cuda']:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")

        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        if forward_info['data_layout'] == 'channels_last':
            img = img.to(memory_format=torch.channels_last)
    else:
        img = data[0]
        with torch.no_grad():
            if not forward_info['no_cuda']:
                img = img.cuda()
            img.sub_(forward_info['mean']).div_(forward_info['std'])

    ploc, plabel = model(img)

    ploc, plabel = ploc.float(), plabel.float()
    trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
    gloc = Variable(trans_bbox, requires_grad=False)
    glabel = Variable(label, requires_grad=False)

    return ploc, plabel, gloc, glabel

def post_process(result, data, forward_info):
    inv_map = {v: k for k, v in forward_info['val_dataset'].label_map.items()}
    ret = []
    ploc, plabel, _, _ = result
    img, img_id, img_size, _, _ = data
    ploc, plabel = ploc.float(), plabel.float()
    # Handle the batch of predictions produced
    # This is slow, but consistent with old implementation.
    for idx in range(ploc.shape[0]):
        # ease-of-use for specific predictions
        ploc_i = ploc[idx, :, :].unsqueeze(0)
        plabel_i = plabel[idx, :, :].unsqueeze(0)

        try:
            result = forward_info['encoder'].decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
        except Exception as e:
            print("Skipping idx {}, failed to decode with message {}, Skipping.".format(idx, e))
            continue

        htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
        loc, label, prob = [r.cpu().numpy() for r in result]
        for loc_, label_, prob_ in zip(loc, label, prob):
            ret.append([img_id[idx], loc_[0] * wtot, \
                        loc_[1] * htot,
                        (loc_[2] - loc_[0]) * wtot,
                        (loc_[3] - loc_[1]) * htot,
                        prob_,
                        inv_map[label_]])
    return ret

def eval_func(final_results, forward_info):
    cocoDt = forward_info['cocoGt'].loadRes(final_results, use_ext=True)

    E = COCOeval(forward_info['cocoGt'], cocoDt, iouType='bbox', use_ext=True)
    E.evaluate()
    E.accumulate()
    if forward_info['local_rank'] == 0:
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E.summarize()

    return E.stats[0]