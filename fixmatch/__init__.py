import torch
import torch.nn as nn
from tqdm import tqdm

def train(net, labeled_loader, unlabeled_loader, train_optimizer, threshold, lambda_u):
    net.train()
    CrossEntropyLoss = nn.CrossEntropyLoss()
    SoftMax = nn.Softmax()

    data_loader = zip(labeled_loader, unlabeled_loader)
    total_loss, total_num, total_correct, total_accepted, train_bar = 0.0, 0, 0, 0, tqdm(data_loader)


    for labeled_batch, unlabeled_batch in train_bar:
        x_l , y_l = labeled_batch[0].to('cuda'),labeled_batch[1].to('cuda')
        xW_u, xS_u, y_u = unlabeled_batch[0].to('cuda'),unlabeled_batch[1].to('cuda'),unlabeled_batch[2].to('cuda')

        # Maybe we could cat the tensors then chunk the result
        logits_l = net(x_l)
        logitsW_u = net(xW_u)
        logitsS_u = net(xS_u)

        loss_l = CrossEntropyLoss(logits_l,y_l)

        # I believe we need to stop the gradient here
        predicitionW_u = SoftMax(logitsW_u).detach()
        mask = predicitionW_u.ge(threshold)

        indices = mask.sum(dim=1).nonzero().flatten()
        unlabeled_samples_accepted = len(indices)
        pseudolabel = predicitionW_u.argmax(dim=1)

        if unlabeled_samples_accepted > 0:
            loss_u = CrossEntropyLoss(logitsS_u[indices,:], pseudolabel[indices])
        else:
            loss_u = 0

        loss = loss_l + lambda_u * loss_u

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()


        correct_pseudolabels = (pseudolabel[indices] == y_u[indices]).sum()

        total_samples = y_l.size()[0] + unlabeled_samples_accepted

        total_correct += correct_pseudolabels
        total_accepted += unlabeled_samples_accepted

        total_num += total_samples
        total_loss += loss.item() * total_samples

        train_bar.set_description('Loss: {:.4f}, Unlabeled samples: {}, Accuracy of pseudolabels: {:.4f}'.format(total_loss/total_num, unlabeled_samples_accepted, total_correct/(total_accepted+1**-20)))

    return total_loss/total_num, total_correct/(total_accepted+1**-20)
