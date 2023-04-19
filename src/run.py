import gc
import argparse

import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from braindecode.models import EEGNetv4
from braindecode.util import np_to_th

from double_se_eegnet import DoubleSEEEGNet
from utils import get_bnci_data, TrainObject, get_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", dest="proposed_model", action="store_false")
    parser.set_defaults(proposed_model=True)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.RandomState(seed)

    X_bnci, y_bnci = get_bnci_data()
    
    in_chans = 22
    labelsize = 4
    window_size = 500

    batch_size = 500
    n_epoch = 500
    n_early_stopping = 50

    highest_accs = []
    for subj in range(9):
        train_subjs = [item for item in list(range(9)) if item not in [subj]]
        train_set = TrainObject(np.vstack(X_bnci[train_subjs]), y=np.concatenate(y_bnci[train_subjs]))
        test_set = TrainObject(X_bnci[subj], y=y_bnci[subj])

        print(f'SUBJECT {subj} START')

        savename = f'subj{subj}.pth'
        Tlosses, Taccuracies = [], []
        Vlosses, Vaccuracies = [], []

        total_epoch = -1
        highest_acc = 0

        if args.proposed_model:
            model = DoubleSEEEGNet(in_chans, labelsize, input_window_samples=window_size, final_conv_length='auto')
        else:
            model = EEGNetv4(in_chans, labelsize, input_window_samples=window_size, final_conv_length='auto')
        
        if cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters())

        cnt = 0
        last_epoch = 0
        for i_epoch in range(n_epoch):
            last_epoch = i_epoch
            if cnt > n_early_stopping:
                print('Early Stopping\n')
                last_epoch = i_epoch-1
                break
            
            total_epoch += 1

            valid_true, valid_pred = [], []
            i_trials_in_batch = get_batches(len(train_set.X), rng, shuffle=True, batch_size=batch_size)

            model.train()
            for i_trials in i_trials_in_batch:
                batch_X = train_set.X[i_trials][:, :, :, None] # Add empty 4th dimension for training
                batch_y = train_set.y[i_trials]

                net_in = np_to_th(batch_X)
                if cuda:
                    net_in = net_in.cuda()

                net_target = np_to_th(batch_y)
                if cuda:
                    net_target = net_target.cuda()

                optimizer.zero_grad()

                outputs = model(net_in)

                loss = F.nll_loss(outputs, net_target)
                loss.backward()

                optimizer.step()

            with torch.no_grad():
                model.eval()
                for setname, dataset in (('Train', train_set), ('Valid', test_set)):
                    i_trials_in_batch = get_batches(len(dataset.X), rng, shuffle=False, batch_size=batch_size)
                    outputs = None

                    for i_trials in i_trials_in_batch:
                        batch_X = dataset.X[i_trials][:, :, :, None]
                        batch_y = dataset.y[i_trials]

                        net_in = np_to_th(batch_X)
                        if cuda:
                            net_in = net_in.cuda()

                        toutputs = model(net_in)
                        if outputs is None:
                            temp = toutputs.cpu()
                            outputs = temp.detach().numpy()
                        else:
                            temp = toutputs.cpu()
                            outputs = np.concatenate((outputs,temp.detach().numpy()))

                    net_target = np_to_th(dataset.y)

                    loss = F.nll_loss(torch.from_numpy(outputs), net_target)

                    predicted_labels = np.argmax((outputs), axis=1)
                    accuracy = np.mean(dataset.y  == predicted_labels)

                    if setname == 'Train':
                        Tlosses.append(loss)
                        Taccuracies.append(accuracy)
                        current_Tacc=accuracy
                    elif setname == 'Valid':
                        Vlosses.append(loss)
                        Vaccuracies.append(accuracy)
                        if accuracy >= highest_acc:
                            cnt = 0
                            print("{:6s} Accuracy: {:.2f}%".format(setname, accuracy * 100))
                            torch.save({
                                'in_chans': in_chans,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'n_classes': 4,
                                'input_time_length': window_size
                            }, savename)

                            highest_acc = accuracy
                        else:
                            cnt += 1
            
        train_loss = np.mean(Tlosses)
        valid_loss = np.mean(Vlosses)

        t = np.arange(0.0, len(Tlosses), 1)+1
        plt.plot(t, Tlosses, 'r', t, Vlosses, 'y')
        plt.legend(('Training loss', 'validation loss'))
        plt.show()

        plt.plot(t, Taccuracies, 'r', t, Vaccuracies, 'y')
        plt.legend(('Training accuracy', 'Validation accuracy'))
        plt.show()
        
        print(f'\nSUBJECT {subj} METRICS')
        print(f'Total Epoch: {last_epoch}')
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Valid Loss: {valid_loss:.6f}")
        print(f"Valid Accuracy: {(highest_acc*100):.2f}%")

        highest_accs.append(highest_acc)

        del model
        gc.collect()
        torch.cuda.empty_cache()


    print(f'Average: {np.mean(highest_accs)*100:.2f}%')
    for hacc in highest_accs:
        print(f'{hacc:.6f}')
