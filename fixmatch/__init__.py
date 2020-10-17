import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm

def save_weights(weight_dir, epoch, loss, net, optimizer, ema):
    if weight_dir is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net,
            'optimizer_state_dict': optimizer,
            'loss': loss,
        }, os.path.join(weight_dir, f'net-{epoch:05d}.pt'))
        torch.save(ema, os.path.join(weight_dir, f'ema-{epoch:05d}.pt'))

def train(net, labeled_dataloader, unlabeled_dataloader, validation_dataloader, optimizer, threshold, lambda_u, epochs, ema_model, device, log_file=None, weight_dir=None):
    net.train()
    net.to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
    SoftMax = nn.Softmax(dim=1)

    if unlabeled_dataloader is None:
        unlabeled_dataloader = [None] * len(labeled_dataloader)

    # Save values for cosine learning rate decay
    initial_lr = optimizer.param_groups[0]['lr']
    total_steps = min(len(labeled_dataloader), len(unlabeled_dataloader)) * epochs
    current_step = 0

    # Write header to log file and save initial weights
    if log_file is not None:
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_acc,pseudo_acc,accepted\n')
    save_weights(weight_dir=weight_dir,
                 epoch=0,
                 loss=-1,
                 net=net.state_dict(),
                 optimizer=optimizer.state_dict(),
                 ema=ema_model.emamodel.state_dict())

    for epoch in range(1, epochs + 1):
        print('Current epoch:', epoch)

        data_loader = zip(labeled_dataloader, unlabeled_dataloader)
        total_loss, total_num, total_correct, total_accepted, train_bar = 0.0, 0, 0, 0, tqdm(data_loader)

        for labeled_batch, unlabeled_batch in train_bar:
            x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
            if unlabeled_batch is not None:
                xW_u, xS_u, y_u = unlabeled_batch[0].to(device), unlabeled_batch[1].to(device), unlabeled_batch[2].to(device)

            # Maybe we could cat the tensors then chunk the result
            logits_l = net(x_l)
            if unlabeled_batch is not None:
                logitsW_u = net(xW_u)
                logitsS_u = net(xS_u)

            loss_l = CrossEntropyLoss(logits_l, y_l)

            loss_u, unlabeled_samples_accepted, correct_pseudolabels = 0, 0, 0
            if unlabeled_batch is not None:
                # I believe we need to stop the gradient here
                predicitionW_u = SoftMax(logitsW_u).detach()
                mask = predicitionW_u.ge(threshold)

                indices = mask.sum(dim=1).nonzero().flatten()
                unlabeled_samples_accepted = len(indices)
                pseudolabel = predicitionW_u.argmax(dim=1)

                if unlabeled_samples_accepted > 0:
                    loss_u = nn.CrossEntropyLoss(logitsS_u[indices,:], pseudolabel[indices],reduction='sum')/y_u.size()[0]

                correct_pseudolabels = (pseudolabel[indices] == y_u[indices]).sum()

            loss = loss_l + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.update(net)

            total_samples = y_l.size()[0] + y_u.size()[0]

            total_correct += correct_pseudolabels
            total_accepted += unlabeled_samples_accepted

            total_num += total_samples
            total_loss += loss.item() * total_samples

            # Cosine learning rate decay
            lr = initial_lr * math.cos(7 * math.pi * current_step / (16 * total_steps))
            for group in optimizer.param_groups:
                group['lr'] = lr
            current_step += 1

            train_bar.set_description('Train loss: {:.4f}, Unlabeled samples: {}, Accuracy of pseudolabels: {:.4f}, LR: {:.4f}'.format(total_loss/total_num, unlabeled_samples_accepted, total_correct/(total_accepted+1e-20), lr))

        # Save logs and weights
        if (epoch == 1 or epoch % 10 == 0 or epoch == epochs) and weight_dir is not None:
            save_weights(weight_dir=weight_dir,
                         epoch=epoch,
                         loss=total_loss / total_num,
                         net=net.state_dict(),
                         optimizer=optimizer.state_dict(),
                         ema=ema_model.emamodel.state_dict())
            # Calculate val_loss/acc
            net.eval()
            with torch.no_grad():
                total_val, correct_val, loss_val = 0, 0, 0.0
                for x_val, y_val in validation_dataloader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits_val = net(x_val)
                    loss_val += CrossEntropyLoss(logits_val, y_val).item() * y_val.size(0)
                    _, predicted_val = torch.max(logits_val, 1)
                    total_val += y_val.size(0)
                    correct_val += (predicted_val == y_val).sum().item()
            net.train()
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(f'{epoch},{total_loss / total_num:.4f},{loss_val / total_val:.4f},{correct_val / total_val:.4f},{total_correct/(total_accepted+1e-20):.4f},{total_accepted}\n')


    return total_loss/total_num, total_correct/(total_accepted+1e-20)
