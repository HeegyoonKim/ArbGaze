import os, shutil
import torch
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, base_lr, epoch, lr_decay_epoch):
    lr = base_lr * (0.1**(epoch//lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_teacher_path(args):
    if args.baseline == 'resnet18':
        teacher_path = './results/%s/HR/f%d/saved_models' % (args.dataset_name, args.fold)
    else:
        teacher_path = './results/%s/HR/resnet%d/f%d/saved_models' % (args.dataset_name, args.baseline, args.fold)
    teacher_file = sorted(os.listdir(teacher_path))[-1]
    teacher_path = os.path.join(teacher_path, teacher_file)
    
    return teacher_path


def plot_loss(save_path, dataset_name):
    epoch_list, test_loss_list = read_loss(save_path, dataset_name)

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('Mean gaze error (degree)')
    plt.grid(True)

    plt.plot(epoch_list, test_loss_list, 'go--', label=dataset_name)

    # Annotation
    for i in range(len(epoch_list)):
        plt.annotate(str(test_loss_list[i]), xy=(epoch_list[i],test_loss_list[i]), ha='center', fontsize=6)
    
    save_path = save_path
    plt.title('%s' % save_path.split('/')[-1])
    plt.legend()
    plt.savefig(save_path + '/loss_curve.png', dpi=200)


def prepare_path(path):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    return path


def read_loss(save_path, tag):
    path = save_path + '/losses_%s.txt' % tag
    f = open(path, 'r')
    lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].split()

    epoch_list = []
    loss_list = []
    for i in range(len(lines)):
        epoch_list.append(int(lines[i][1].lstrip('[').rstrip(']')))
        loss_list.append(float(lines[i][9].lstrip('(').rstrip(')')))

    return epoch_list, loss_list


def remove_checkpoints(save_path):
    save_path = os.path.join(save_path, 'saved_models')
    saved_models = sorted(os.listdir(save_path))[:-1]
    for saved_model in saved_models:
        os.remove(os.path.join(save_path, saved_model))


def save_args(args):
    file_name = os.path.join(args.save_path, 'arguments.txt')
    with open(file_name, 'w') as f:
        for n, v in args.__dict__.items():
            f.write('{0}\n{1}\n\n'.format(n, v))


def save_checkpoint(state, is_best, save_path):
    save_path = os.path.join(save_path, 'saved_models')
    save_path = prepare_path(save_path)
    file_name = '%d.pth' % (state['epoch'])
    best_file_name = 'best_' + file_name
    file_path = os.path.join(save_path, file_name)
    best_file_path = os.path.join(save_path, best_file_name)
    torch.save(state, file_path)
    # Remove previous best model
    if is_best:
        saved_models = os.listdir(save_path)
        for saved_model in saved_models:
            if saved_model.startswith('best'):
                os.remove(os.path.join(save_path, saved_model))
        shutil.copyfile(file_path, best_file_path)


def send_data_dict_to_gpu(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device)
        elif isinstance(v, list):
            data_list = []
            for i in range(len(v)):
                data_list.append(v[i].detach().to(device))
            data[k] = data_list
    return data


def train_and_test_split(dataset_name, fold):
    if dataset_name == 'utmv':
        subject_group_1 = [s for s in range(17)]
        subject_group_2 = [s for s in range(17, 34)]
        subject_group_3 = [s for s in range(34, 50)]
        if fold == 1:
            TRAIN_SUBJECT_IDS = subject_group_2 + subject_group_3
            TEST_SUBJECT_IDS = subject_group_1
        elif fold == 2:
            TRAIN_SUBJECT_IDS = subject_group_3 + subject_group_1
            TEST_SUBJECT_IDS = subject_group_2
        else:   # fold == 3
            TRAIN_SUBJECT_IDS = subject_group_1 + subject_group_2
            TEST_SUBJECT_IDS = subject_group_3
    else: # mpiigaze
        all_subjects = [s for s in range(15)]
        all_subjects.remove(fold)
        TRAIN_SUBJECT_IDS = all_subjects
        TEST_SUBJECT_IDS = [fold]
    
    return TRAIN_SUBJECT_IDS, TEST_SUBJECT_IDS


class AverageMeter(object):
    def __init__(self):
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
