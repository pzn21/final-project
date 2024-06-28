import torch
import numpy as np
from parser import args
from model import generator, discriminator
from dataset import ImageDataset
import torch.utils.data as data
import imageio.v2 as imageio
import torch.nn as nn


def train(args, gen, dis, dataloader, opt_g, opt_d, num_epoch):
    gen.train()
    dis.train()
    gen_losses = []
    dis_losses = []
    for i, batch in enumerate(dataloader):
        img, sketch = batch[0].to(args.device), batch[1].to(args.device)
        g_img = gen(sketch)
        d_value_2 = dis(g_img, sketch)
        g_loss = - torch.log(d_value_2).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        gen_losses.append(g_loss.item())
        g_img = gen(sketch)
        d_value_1 = dis(img, sketch)
        d_value_2 = dis(g_img, sketch)
        d_loss = - (torch.log(d_value_1) + torch.log(1 - d_value_2)).mean()
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        dis_losses.append(d_loss.item())
        if i % 10 == 0:
            print('epoch:', num_epoch, 'batch:', i, 'gen loss:', g_loss.item(), 'dis loss:', d_loss.item())
    return np.mean(gen_losses), np.mean(dis_losses)


@torch.no_grad()
def test(args, gen, dis, dataloader, num_epoch):
    gen.eval()
    dis.eval()
    gen_losses = []
    dis_losses = []
    for i, batch in enumerate(dataloader):
        img, sketch = batch[0].to(args.device), batch[1].to(args.device)
        g_img = gen(sketch)
        d_value_2 = dis(g_img, sketch)
        g_loss = - torch.log(d_value_2).mean()
        gen_losses.append(g_loss.item())
        d_value_1 = dis(img, sketch)
        d_loss = - (torch.log(d_value_1) + torch.log(1 - d_value_2)).mean()
        dis_losses.append(d_loss.item())
        if i < 3:
            example_img = ((g_img[0] + 1.0) * 127.5).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            imageio.imwrite('result/epoch{}_example{}_img.png'.format(num_epoch, i), example_img)
            gt_img = ((img[0] + 1.0) * 127.5).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            imageio.imwrite('result/epoch{}_example{}_gt.png'.format(num_epoch, i), gt_img)
            example_sketch = (sketch[0] * 255).cpu().numpy()[0].astype(np.uint8)
            imageio.imwrite('result/epoch{}_example{}_sketch.png'.format(num_epoch, i), example_sketch)
    return np.mean(gen_losses), np.mean(dis_losses)


if __name__ == '__main__':
    dataset = ImageDataset(args)
    train_ratio = 0.8
    train_len = int(len(dataset) * train_ratio)
    whole_dataset = data.random_split(dataset, [train_len, len(dataset) - train_len])
    train_dataset, test_dataset = whole_dataset[0], whole_dataset[1]
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=8)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                      drop_last=True, num_workers=8)

    gen = nn.DataParallel(generator(args).to(args.device))
    dis = nn.DataParallel(discriminator().to(args.device))
    optimizer_g = torch.optim.Adam(gen.parameters(), lr=args.lr_gen, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.Adam(dis.parameters(), lr=args.lr_dis, weight_decay=0)
    for epoch in range(args.epochs):
        print('Start epoch{}'.format(epoch))
        print('Start training')
        gen_loss, dis_loss = train(args, gen, dis, train_dataloader, optimizer_g, optimizer_d, epoch)
        print('gen loss:', gen_loss, 'dis loss:', dis_loss)
        print('Start testing')
        gen_loss, dis_loss = test(args, gen, dis, train_dataloader, epoch)
        print('gen loss:', gen_loss, 'dis loss:', dis_loss)
