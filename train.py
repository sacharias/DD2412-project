import torch.nn as nn
import torch
from tqdm import tqdm

def train(net, labeled_loader, unlabeled_loader, train_optimizer, threshold, lambda_u):
    net.train()
    CrossEntropyLoss = nn.CrossEntropyLoss()
    SoftMax = nn.Softmax()

    data_loader = zip(labeled_loader, unlabeled_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)


    for labeled_batch, unlabeled_batch in train_bar:
        x_l , y_l = labeled_batch
        xW_u, xS_u, y_u = unlabeled_batch


        #Maybe we could stack the tensor then unstack the result
        logits_l  = net(x_l)
        logitsW_u  = net(xW_u)
        logitsS_u  = net(xS_u)

        loss_l = CrossEntropyLoss(logits_l,y_l)

        #I believe we need to stop the gradient here
        predicitionW_u = SoftMax(logitsW_u).detach()
        mask = predicitionW_u.ge(threshold)
        """
        Either apply the mask to the samples then calculate loss, or we calculate the loss then apply the mask.
        Not sure which one is faster however probably dependend on how much time it takes to create the new tensors when applying the mask.
        """
        pseudolabel = predicitionW_u.argmax(dim=1)

        #This becomes NaN when all elements get masked
        loss_u = CrossEntropyLoss(logitsS_u, pseudolabel).masked_select(mask).mean()
        loss_u[torch.isnan(loss_u)] = 0

        loss = loss_l + lambda_u * loss_u
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()


        total_samples = y_l.size()[0] + y_u.size()[0]

        total_num += total_samples
        total_loss += loss.item() * total_samples
        train_bar.set_description('Loss: {:.4f}'.format(total_loss/total_num))

    return total_loss/total_num
