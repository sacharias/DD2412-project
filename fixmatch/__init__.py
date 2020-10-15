import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm

def train(net, labeled_loader, unlabeled_loader, train_optimizer, threshold, lambda_u, epochs, ema_model, device, log_file=None, history_dir=None):
    net.train()
    net.to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
    SoftMax = nn.Softmax(dim=1)

    # Save values for cosine learning rate decay
    initial_lr = train_optimizer.param_groups[0]['lr']
    total_steps = min(len(labeled_loader), len(unlabeled_loader)) * epochs
    current_step = 0

    # Write header to log file
    if log_file is not None:
        log_file.write('loss,pseudoacc\n')

    for epoch in range(1, epochs + 1):
        print('Current epoch:', epoch)

        data_loader = zip(labeled_loader, unlabeled_loader)
        total_loss, total_num, total_correct, total_accepted, train_bar = 0.0, 0, 0, 0, tqdm(data_loader)

        for labeled_batch, unlabeled_batch in train_bar:
            x_l, y_l = labeled_batch[0].to(device),labeled_batch[1].to(device)
            xW_u, xS_u, y_u = unlabeled_batch[0].to(device),unlabeled_batch[1].to(device),unlabeled_batch[2].to(device)

            # Maybe we could cat the tensors then chunk the result
            logits_l = net(x_l)
            logitsW_u = net(xW_u)
            logitsS_u = net(xS_u)

            loss_l = CrossEntropyLoss(logits_l, y_l)

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

            ema_model.update(net)

            correct_pseudolabels = (pseudolabel[indices] == y_u[indices]).sum()

            total_samples = y_l.size()[0] + unlabeled_samples_accepted

            total_correct += correct_pseudolabels
            total_accepted += unlabeled_samples_accepted

            total_num += total_samples
            total_loss += loss.item() * total_samples

            # Cosine learning rate decay
            lr = initial_lr * math.cos(7 * math.pi * current_step / (16 * total_steps))
            for group in train_optimizer.param_groups:
                group['lr'] = lr
            current_step += 1

            train_bar.set_description('Loss: {:.4f}, Unlabeled samples: {}, Accuracy of pseudolabels: {:.4f}, LR: {:.4f}'.format(total_loss/total_num, unlabeled_samples_accepted, total_correct/(total_accepted+1e-20), lr))

        # Save logs and weights
        if log_file is not None:
            log_file.write(f'{total_loss/total_num},{total_correct/(total_accepted+1e-20)}\n')
        if history_dir is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': train_optimizer.state_dict(),
                'loss': total_loss/total_num,
            }, os.path.join(history_dir, f'epoch-{epoch}.pt'))

    return total_loss/total_num, total_correct/(total_accepted+1e-20)
