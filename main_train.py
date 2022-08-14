from makedataset import makeDataset
from model import UNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Losses import DiceLoss, GeneralizedDiceLoss
from torchvision import transforms


class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.normalization = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.normalization(inputs)

        targets = targets[:, 1:2, ...]
        inputs = torch.where(inputs[:, 1:2, ...] > 0.5, 1.0, 0.0)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


# define Transform
tr = transforms.Compose([
    transforms.RandomCrop(256)
])

# make dataLoader
trainds = makeDataset(kind='train', location='data_npy_2ch')
validds = makeDataset(kind='valid', location='data_npy_2ch')

trainLoader = DataLoader(trainds, batch_size=config.BATCH_SIZE, shuffle=True,
                         pin_memory=config.PIN_MEMORY)
validLoader = DataLoader(validds, batch_size=config.BATCH_SIZE, shuffle=False,
                         pin_memory=config.PIN_MEMORY)

params = [0.0001]
for (lr_) in params:
    # Define Model################################################################################################
    unet = UNet(64, 5, use_xavier=True, use_batchNorm=True, dropout=0.5, retain_size=True, nbCls=2)

    devices = 'cpu'
    device_num = 0
    if torch.cuda.is_available():
        devices = 'gpu'
        device_num = torch.cuda.device_count()
        if device_num > 1:
            unet = torch.nn.DataParallel(unet)
    unet.to(config.DEVICE)
    #############################################################################################################

    # Define History, optimizer, schedular, loss function########################################################
    history = {'train_loss': [], 'valid_loss': [], 'dice_valid_score': []}
    num_train = int(len(trainds) // config.BATCH_SIZE)
    writer = SummaryWriter(log_dir='./runs/Train')
    opt = torch.optim.NAdam(unet.parameters(), lr=lr_)
    schedular = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.25, verbose=True)
    dicelossfunc = GeneralizedDiceLoss(normalization='softmax')
    diceScore = DiceScore()
    #############################################################################################################

    # main train#################################################################################################
    pbar = tqdm(range(config.N_EPOCHS), leave=False, position=0)
    for e in pbar:
        unet.train()
        totalloss = 0
        totalvalidloss = 0
        totalvaliddice = 0

        trainstep = 0
        validstep = 0

        inner_pbar = tqdm(range(num_train), leave=False, position=1)
        data_iter = iter(trainLoader)
        for i in inner_pbar:
            (x, y) = next(data_iter)
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            pred = unet(x)
            loss = dicelossfunc(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            totalloss += loss
            trainstep += 1
            inner_pbar.set_postfix({'Train_loss': "{:.4f}".format(loss)})

        with torch.no_grad():
            unet.eval()
            for (x, y) in validLoader:
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                pred = unet(x)
                validloss = dicelossfunc(pred.clone(), y.clone())
                totalvalidloss += (validloss)

                validScore = diceScore(pred, y)

                totalvaliddice += validScore
                validstep += 1

        avgloss = (totalloss / trainstep).cpu().detach().numpy()
        avgvalidloss = (totalvalidloss / validstep).cpu().detach().numpy()
        avgvaliddice = (totalvaliddice / validstep).cpu().detach().numpy()

        schedular.step(avgvalidloss)

        history['train_loss'].append(avgloss)
        history['valid_loss'].append(avgvalidloss)
        history['dice_valid_score'].append(avgvaliddice)

        writer.add_scalar('train_loss', avgloss, e)
        writer.add_scalar('validation_loss', avgvalidloss, e)
        writer.add_scalar('validation_dice', avgvaliddice, e)

        writer.add_scalars('loss', {'Train': avgloss, 'Valid': avgvalidloss}, e)

        pbar.set_postfix({'Train_avg_loss': '{:.4f}'.format((avgloss)),
                          'Valid_avg_loss': '{:.4f}'.format((avgvalidloss)),
                          'Valid_avg_dice': '{:.4f}%'.format(100 * avgvaliddice)})

        torch.save(unet.state_dict(), './final_result/unet_{}.pt'.format(e + 1))
        with open('./final_result/history_{}.pkl'.format(e + 1), 'wb') as f:
            pickle.dump(history, f)

    writer.flush()
    writer.close()

    print('Saving model...\n\n')
    torch.save(unet.state_dict(), './final_result/UNet.pt')

    print('Saving figure...\n\n')
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 10))
    plt.plot(history['train_loss'], label='Train_Dice_Loss')
    plt.plot(history['valid_loss'], label='Validation_Dice_Loss')
    plt.title('Training Dice Score on Dataset')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Dice Loss')
    plt.legend(loc='lower left')
    plt.savefig('./final_result/train_result.png')

    print('Saving History...\n\n')
    with open('./final_result/history.pkl', 'wb') as f:
        pickle.dump(history, f)

print('***************End of System***************')
