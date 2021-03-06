import numpy as np
import src.utils as utils
import src.stat as stats
from tqdm import tqdm
from torch.autograd import Variable


def test(test_loader, model, criterion, stat, predict=False):
    losses = utils.AverageMeter()
    model.eval()

    all_dist, all_output, all_target, all_input, all_bool = [], [], [], [], []

    for i, (inps, tars, good_keypts, keys) in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        #make prediction with model
        outputs = model(inputs)
        all_output.append(outputs.data.cpu().numpy())
        all_input.append(inputs.data.cpu().numpy())
        
        if not predict:
            # calculate loss
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
        
            outputs[~good_keypts] = 0
            targets[~good_keypts] = 0

            # undo normalisation to calculate accuracy in real units
            dim=1
            dimensions = stat['targets_1d']
            out = stats.unNormalize(outputs.data.cpu().numpy(), stat['mean'][dimensions], stat['std'][dimensions])
            tar = stats.unNormalize(targets.data.cpu().numpy(), stat['mean'][dimensions], stat['std'][dimensions])
            abserr = np.abs(out - tar)

            n_pts = len(dimensions)//dim
            distance = np.zeros_like(abserr)
            for k in range(n_pts):
                distance[:, k] = np.sqrt(np.sum(abserr[:, dim*k:dim*(k + 1)], axis=1))

            all_dist.append(distance)
            all_target.append(targets.data.cpu().numpy())
            all_bool.append(good_keypts)
       
    all_input = np.vstack(all_input)
    all_output = np.vstack(all_output)
    
    if predict:
        return None, None, None, None, all_output, None, all_input, None
        
    all_target = np.vstack(all_target)
    all_dist = np.vstack(all_dist)
    all_bool = np.vstack(all_bool)
    
    #mean errors
    all_dist[all_dist == 0] = np.nan
    joint_err = np.nanmean(all_dist, axis=0)
    ttl_err = np.nanmean(joint_err)

    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err, joint_err, all_dist, all_output, all_target, all_input, all_bool