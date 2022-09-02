import argparse, random, logging, os
import numpy as np

import torch
from torch.utils.data import DataLoader

from networks.ArbGaze import ArbGaze
from data import GazeData
from losses import GazeAngularLoss, KDLoss
from utils import *


parser = argparse.ArgumentParser()
# Save
parser.add_argument('--save_path', type=str, default='./results')
# Data
parser.add_argument('--data_dir', type=str, default='../datasets')
parser.add_argument('--dataset_name', type=str, choices=['utmv', 'mpii'])
parser.add_argument('--fold', type=int, choices=[i for i in range(15)],
                    help='1~3 for 3-fold cross validation on utmv, 0~14 for leave-one-out on mpii')
parser.add_argument('--train_img_type', type=str, default='BC', choices=['BC', 'LR', 'HR'])
parser.add_argument('--test_img_type', type=str, default='BC', choices=['BC', 'LR', 'HR'])
parser.add_argument('--max_scale', type=float, default=4.0)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
# Network
parser.add_argument('--backbone_T', type=str, default='resnet18')
parser.add_argument('--backbone_S', type=str, default='resnet18')
parser.add_argument('--num_experts', type=int, default=4)
parser.add_argument('--FA_module', default=False, action='store_true')
parser.add_argument('--which_block', type=str, default='1234', choices=['1234', '12', '34', '1', '2', '3', '4'])
# Training
parser.add_argument('--load_from_teacher', default=False, action='store_true')
parser.add_argument('--base_lr', type=float, default=0.005)
parser.add_argument('--adjust_lr', default=False, action='store_true')
parser.add_argument('--lr_decay_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--training_scale', type=float, default=None, nargs='+')
parser.add_argument('--random_scale_input', default=False, action='store_true')
parser.add_argument('--p_dropout', type=float, default=0.0)
# KD loss function
parser.add_argument('--coeff_KD_loss', type=float, default=10.0)
parser.add_argument('--KD_loss_type', type=str, default=None, choices=['JS', 'COS', 'L1', 'MSE'])
# Validation only
parser.add_argument('--eval_only', default=False, action='store_true')
parser.add_argument('--eval_scales', type=float, nargs='+',
                    help='if None, all scale factors')
args = parser.parse_args()


# Basic settings
_ = prepare_path(args.save_path)
save_args(args)
logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Scale factors for training
if args.training_scale is not None:    # Multiple gaze baseline
    SCALES = args.training_scale
else:
    SCALES = np.arange(1.1, 4.1, 0.1)
# Scale factors for evaluation
if args.eval_scales is None:
    args.eval_scales = np.arange(1.0, 4.1, 0.1)

# Data setting
DATA_PATH = args.data_dir + '/%s.h5' % args.dataset_name
# 3-fold for utmv, leave-one-out for mpii
TRAIN_SUBJECT_IDS, TEST_SUBJECT_IDS = train_and_test_split(args.dataset_name, args.fold)
logging.info('Dataset: %s' % args.dataset_name)
logging.info('Train  : {}'.format(TRAIN_SUBJECT_IDS))
logging.info('Test   : {}'.format(TEST_SUBJECT_IDS))
logging.info('')


def main():
    # Reproducibility
    seed_value = 100
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed_value)
    
    pretrained_path_for_T = None
    pretrained_path_for_S = None

    if args.load_from_teacher:
        pretrained_path_for_S = get_pretrained_path(args, args.backbone_S)
    
    if args.KD_loss_type is not None:
        pretrained_path_for_T = get_pretrained_path(args, args.backbone_T)
    
    logging.info('Teacher  : %s' % args.backbone_T)
    logging.info('Load from: %s' % pretrained_path_for_T)
    logging.info('Student  : %s' % args.backbone_S)
    logging.info('Load from: %s' % pretrained_path_for_S)
    logging.info('')

    epoch = 0
    min_error = 1e10
    if args.eval_only:
        args.num_epochs = 0
    
    # Student and teacher network
    teacher_net = ArbGaze(
        backbone=args.backbone_T,
        FA_module=False,
        num_experts=0,
        pretrained_path=pretrained_path_for_T
    ).to(DEVICE)
    student_net = ArbGaze(
        backbone=args.backbone_S,
        FA_module=args.FA_module,
        num_experts=args.num_experts,
        pretrained_path=pretrained_path_for_S
    ).to(DEVICE)

    # Dataset & dataloader
    train_dataset = GazeData(dataset_name=args.dataset_name, data_path=DATA_PATH, subject_ids=TRAIN_SUBJECT_IDS)
    test_dataset = GazeData(dataset_name=args.dataset_name, data_path=DATA_PATH, subject_ids=TEST_SUBJECT_IDS)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Loss function
    GZ_loss_function = GazeAngularLoss()
    KD_loss_function = KDLoss(args.KD_loss_type)
    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_net.parameters()), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-8)


    for epoch in range(epoch, args.num_epochs):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        np.random.seed(epoch)

        if args.adjust_lr:
            adjust_learning_rate(optimizer, args.base_lr, epoch, args.lr_decay_epoch)
        
        # Train one epoch
        train(train_dataset, student_net, teacher_net, GZ_loss_function, KD_loss_function, optimizer, epoch)
        torch.cuda.empty_cache()

        # Evaluation on each dataset
        eval_gaze_error = test(test_loader, student_net, teacher_net, GZ_loss_function, KD_loss_function, epoch, args.max_scale, write_error=True)

        is_best = eval_gaze_error < min_error
        min_error = min(eval_gaze_error, min_error)
        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': student_net.state_dict()},
            is_best, args.save_path
        )
    
    if args.train_img_type == 'HR':
        args.test_img_type = 'BC'
    
    if args.eval_only:
        eval_over_scales(test_dataset, student_net, teacher_net, GZ_loss_function, KD_loss_function)
    else:
        remove_checkpoints(args.save_path)    
        plot_loss(args.save_path, args.dataset_name)
        eval_over_scales(test_dataset, student_net, teacher_net, GZ_loss_function, KD_loss_function)


def train(train_dataset, student_net, teacher_net, GZ_loss_function, KD_loss_function, optimizer, epoch):
    logging.info('Train.......')

    if (args.train_img_type == 'HR') or (args.training_scale is not None):
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        scale_idx = np.random.randint(0, 30)
        train_dataset.set_scale(SCALES[scale_idx])

    loss_file = os.path.join(args.save_path, 'losses_train.txt')
    if not os.path.isfile(loss_file):
        write_file = open(loss_file, 'w')
    else:
        write_file = open(loss_file, 'a')
        
    GZ_loss_meter = AverageMeter()
    KD_loss_meter = AverageMeter()

    student_net.train()
    teacher_net.eval()
    for i, (data_dict) in enumerate(loader, 1):
        data_dict = send_data_dict_to_gpu(data_dict, DEVICE)
        batch_size = data_dict['gaze'].size(0)
        scale = data_dict['scale'][0].float().view(1, 1)
        if args.random_scale_input:
            scale = SCALES[np.random.randint(0,30)]
            scale = torch.tensor(scale).to(DEVICE).float().view(1, 1)

        # Extract features from teacher
        with torch.no_grad():
            _, teacher_feats = teacher_net(data_dict['HR_img'], scale)

        # Estimate gaze from student
        gaze_hat, student_feats = student_net(data_dict['%s_img' % args.train_img_type], scale)

        # Calculate loss
        GZ_loss = GZ_loss_function(data_dict['gaze'], gaze_hat)
        loss = GZ_loss
        
        # Add KD loss
        if args.KD_loss_type is not None:
            KD_loss = KD_loss_function(teacher_feats, student_feats)
            loss += args.coeff_KD_loss * KD_loss
                
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        GZ_loss_meter.update(GZ_loss.item(), batch_size)
        if args.KD_loss_type is not None:
            KD_loss_meter.update(KD_loss.item(), batch_size)

        # Print & Write loss
        if i % 50 == 0:
            loss_string = 'Epoch: [%d][%d/%d]  KD Loss: %.4f (%.4f)  GZ Loss: %.3f (%.3f)' % (
                epoch + 1, i, len(loader), KD_loss_meter.val, KD_loss_meter.avg,
                GZ_loss_meter.val, GZ_loss_meter.avg
            )
            logging.info(loss_string)
            write_file.write(loss_string + '\n')
        
        # Train student and not Multiple gaze
        if (args.train_img_type != 'HR') and (args.training_scale is None):
            scale_idx = np.random.randint(0, 30)
            train_dataset.set_scale(SCALES[scale_idx])

    write_file.close()    
    logging.info('')
    return GZ_loss_meter.avg


def test(loader, student_net, teacher_net, GZ_loss_function, KD_loss_function, epoch, scale, write_error=False):
    logging.info('Evaluation on %s dataset......' % args.dataset_name)
    logging.info('Scale: %.2f' % scale)

    GZ_loss_meter = AverageMeter()
    KD_loss_meter = AverageMeter()

    student_net.eval()
    teacher_net.eval()
    for i, (data_dict) in enumerate(loader, 1):
        data_dict = send_data_dict_to_gpu(data_dict, DEVICE)
        scale = data_dict['scale'][0].view(1, 1).float()
        batch_size = data_dict['gaze'].size(0)

        # Estimate gaze
        with torch.no_grad():
            _, teacher_feats = teacher_net(data_dict['HR_img'], scale)
            gaze_hat, student_feats = student_net(data_dict['%s_img' % args.test_img_type], scale)
        
        # Calculate losses
        GZ_loss = GZ_loss_function(data_dict['gaze'], gaze_hat)
        GZ_loss_meter.update(GZ_loss.item(), batch_size)
        if args.KD_loss_type is not None:
            KD_loss = KD_loss_function(teacher_feats, student_feats)
            KD_loss_meter.update(KD_loss.item(), batch_size)            

        # Print process
        if i % 50 == 0:
            process_string = 'Epoch: [%d][%d/%d]' % (epoch + 1, i, len(loader))
            logging.info(process_string)
    
    # Write error during training
    if write_error:
        error_file = os.path.join(args.save_path, 'losses_%s.txt' % args.dataset_name)
        if not os.path.isfile(error_file):
            write_file = open(error_file, 'w')
        else:
            write_file = open(error_file, 'a')
        error_string = 'Epoch: [%d]  KD Loss: %.4f (%.4f)  GZ loss: %.3f (%.3f)' % (
            epoch + 1, KD_loss_meter.val, KD_loss_meter.avg,
            GZ_loss_meter.val, GZ_loss_meter.avg
        )
        logging.info(error_string)
        write_file.write(error_string + '\n')
    
    logging.info('')
    return GZ_loss_meter.avg


def eval_over_scales(dataset, student_net, teacher_net, GZ_loss_function, KD_loss_function):
    logging.info('Evaluation over following scales: {}'.format(args.eval_scales))

    best_model_file = sorted(os.listdir(os.path.join(args.save_path, 'saved_models')))[-1]
    best_model_path = os.path.join(args.save_path, 'saved_models', best_model_file)
    logging.info('Load %s for evaluation\n' % best_model_path)
    saved = torch.load(best_model_path).copy()
    student_net.load_state_dict(saved['state_dict'])
    epoch = saved['epoch']

    for scale in args.eval_scales:
        eval_file = os.path.join(args.save_path, 'evaluation_on_%s.txt' % args.dataset_name)
        if not os.path.isfile(eval_file):
            write_file = open(eval_file, 'w')
        else:
            write_file = open(eval_file, 'a')
        
        dataset.set_scale(scale)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        eval_gaze_error = test(loader, student_net, teacher_net, GZ_loss_function, KD_loss_function, epoch - 1, scale)

        # Write
        write_file.write('Scale: %.2f  Gaze error: %.3f\n' % (scale, eval_gaze_error))
        write_file.close()
        torch.cuda.empty_cache()
    
    logging.info('')


if __name__ == '__main__':
    main()
    print('DONE')