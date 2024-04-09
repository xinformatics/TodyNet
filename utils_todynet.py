import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
              
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        # print(output, target)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
           
            res.append(correct_k.mul_(100.0 / batch_size))
            
        # print(correct_k)
        
        
        ## calculation of auc-roc and auc-pr
        # output_np = torch.softmax(output, dim=1).detach().cpu().numpy()[:,1]
        # print(output_np, target)
        # target_np = target.detach().cpu().numpy()
        # try:
        #     auc_roc_value = roc_auc_score(target_np, output_np)
        # except ValueError:
        #     auc_roc_value = 0
        #     pass
        # auc_roc_value = roc_auc_score(target_np, output_np)
        # auc_pr_value = average_precision_score(target_np, output_np)
        
        # print(auc_roc_value, auc_pr_value)
        
        # return res, auc_roc_value, auc_pr_value
        return res

def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_default_train_val_test_loader(args):

    # get dataset-id
    dsid = args['dataset']

    # get dataset from .pt
    data_train  = torch.load(f'data/UCR/{dsid}/X_train.pt')
    # data_val    = torch.load(f'data/UCR/{dsid}/X_valid.pt')
    data_val    = torch.load(f'data/UCR/{dsid}/X_test.pt')
    
    label_train = torch.load(f'data/UCR/{dsid}/y_train.pt')
    # label_val   = torch.load(f'data/UCR/{dsid}/y_valid.pt')
    label_val   = torch.load(f'data/UCR/{dsid}/y_test.pt')

    label_train = label_train.flatten().to(dtype=torch.int64)
    label_val   = label_val.flatten().to(dtype=torch.int64)
    # init [num_variables, seq_length, num_classes]
    num_nodes = data_val.size(-2)

    seq_length = data_val.size(-1)
    
    num_classes = len(torch.bincount(label_val.type(torch.int)))


    # convert data & labels to TensorDataset
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_val, label_val)

    # data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=args['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args['val_batch_size'],shuffle=False,num_workers=args['workers'],pin_memory=True)


    return train_loader, val_loader, num_nodes, seq_length, num_classes
