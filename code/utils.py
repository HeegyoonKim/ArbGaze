import os, shutil, math
import torch
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, base_lr, epoch, lr_decay_epoch):
    lr = base_lr * (0.1**(epoch//lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_pretrained_path(args, backbone):
    if backbone == 'resnet18':
        pretrained_path = '../results/%s/HR/f%d/saved_models' % (args.dataset_name, args.fold)
    else:
        pretrained_path = '../results/%s/HR/%s/f%d/saved_models' % (args.dataset_name, backbone, args.fold)
    pretrained_file = sorted(os.listdir(pretrained_path))[-1]
    pretrained_path = os.path.join(pretrained_path, pretrained_file)
    
    return pretrained_path


def input_matrix_wpn(inH, inW, scale, add_scale=True):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW = int(scale*inH), int(scale*inW)

    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH,  scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)
    if add_scale:
        scale_mat = torch.zeros(1,1)
        scale_mat[0,0] = 1.0/scale
        scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

    ####projection  coordinate  and caculate the offset 
    h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag,  0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
    mask_mat = mask_mat.eq(2)
    pos_mat = pos_mat.contiguous().view(1, -1,2)
    if add_scale:
        pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

    return pos_mat,mask_mat


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
    else: # mpiifacegaze
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