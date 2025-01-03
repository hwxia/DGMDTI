import torch
import copy
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, \
    precision_score
from model import binary_cross_entropy, cross_entropy_logits
from tqdm import tqdm
from utils import loadESM2andMol


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader,
                 hyperparam_dict, alpha=1):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = hyperparam_dict['MAX_EPOCH']  # 100
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = hyperparam_dict['DECODER_BINARY']  # 1

        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0


    def train(self):

        # 加载预训练模型提取的特征
        esm2_train_list, esm2_val_list, esm2_test_list, mol_train_list, mol_val_list, mol_test_list = (
            loadESM2andMol(len(self.train_dataloader), len(self.val_dataloader), len(self.test_dataloader)))

        print("load...1")

        for i in range(self.epochs):
            self.current_epoch += 1

            # 训练
            train_loss = self.train_epoch(esm2_train_list, mol_train_list)

            # 下面进行验证
            auroc, auprc, val_loss = self.test(esm2_val_list, esm2_test_list,
                                               mol_val_list, mol_test_list,
                                               dataloader="val")
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))

        # 100个epoch后，进行测试
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = (
            self.test(esm2_val_list, esm2_test_list,
                      mol_val_list, mol_test_list,
                      dataloader="test"))
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))


    def train_epoch(self, esm2_train_list, mol_train_list):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)

        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1

            train_esm2_emd = (torch.tensor(esm2_train_list[i])).to(self.device)
            train_mol_emd = (torch.tensor(mol_train_list[i])).to(self.device)
            labels = labels.float().to(self.device)


            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(train_mol_emd, train_esm2_emd)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch


    def test(self, esm2_val_list, esm2_test_list, mol_val_list, mol_test_list, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                labels = labels.float().to(self.device)
                if dataloader == "val":
                    val_esm2_emd = (torch.tensor(esm2_val_list[i])).to(self.device)
                    val_mol_emd = (torch.tensor(mol_val_list[i])).to(self.device)

                    v_d, v_p, f, score = self.model(val_mol_emd, val_esm2_emd)

                elif dataloader == "test":
                    test_esm2_emd = (torch.tensor(esm2_test_list[i])).to(self.device)
                    test_mol_emd = (torch.tensor(mol_test_list[i])).to(self.device)

                    # # 保存最优的model
                    # torch.save(self.model, './model/model_best.pth')
                    # torch.save(self.model.state_dict(), './model/model_best_params.pth')
                    v_d, v_p, f, score = self.best_model(test_mol_emd, test_esm2_emd)

                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
