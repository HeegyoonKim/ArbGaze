import argparse, os, cv2
import numpy as np
import matplotlib.pyplot as plt
from thop import profile

import torch
from torch.utils.data import DataLoader

from networks.ArbGaze import ArbGaze
from networks.MetaSR import MetaRDN
from data import GazeData
from losses import GazeAngularLoss
from utils import (input_matrix_wpn, send_data_dict_to_gpu,
    train_and_test_split, AverageMeter, prepare_path)


parser = argparse.ArgumentParser()
parser.add_argument('--which', type=str, choices=['fig4', 'fig5', 'fig6', 'fig7', 'tab1'], nargs='+')
parser.add_argument('--scales', type=float, nargs='+',
                    help='scale factors on x-axis. if None, all scale factors')
parser.add_argument('--annotate', default=False, action='store_true')
# FLOPs and parameters
parser.add_argument('--utmv_file', type=str, default='../datasets/utmv.h5')
parser.add_argument('--mpii_file', type=str, default='../datasets/mpii.h5')
# Visualize gaze vector
parser.add_argument('--fold', type=int, choices=[i for i in range(15)],
                    help='1~3 for 3-fold cross validation on utmv, 0~14 for leave-one-out on mpii')
parser.add_argument('--vis_scales', type=float, nargs='+')
args = parser.parse_args()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = '../figures'
prepare_path(OUTPUT_DIR)

DATASET_DICT = {
    'utmv': {
        'results_dir': '../results/utmv/',
        'title': 'Mean gaze error over scales on UTMultiview',
        'folds': ['/f%d' % (i+1) for i in range(3)]
    },
    'mpii': {
        'results_dir': '../results/mpii/',
        'title': 'Mean gaze error over scales on MPIIGaze',
        'folds': ['/f%d' % i for i in range(15)]
    }
}

# Scales
if args.scales is None:
    args.scales = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                   2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                   3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def draw_gaze(image_in, center, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(center).astype(np.int32)),
                   tuple(np.round([center[0] + dx,
                                   center[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.4)
    return image_out


def tensor_img_to_np_img(tensor):
    return (tensor.numpy().transpose(0,2,3,1)*255.0).clip(0, 255).astype(np.uint8)


def read_all_evaluations(results, dataset):
    reuslts_dir = DATASET_DICT[dataset]['results_dir']
    folds = DATASET_DICT[dataset]['folds']

    def read_evaluation(result):
        path = reuslts_dir + result + '/evaluation_on_%s.txt' % dataset
        f = open(path, 'r')
        lines = f.readlines()
        # Grab gaze error over scales
        error_list = []
        for i in range(len(lines)):
            scale = float(lines[i][7:10])
            if scale in args.scales:
                error_list.append([scale, float(lines[i][24:-1])])
        f.close()
        error_list = sorted(error_list, key=lambda l:l[0])
        for row in error_list: # Delete scale column
            del row[0]
        error_list = sum(error_list, [])  # 2D to 1D
        return error_list
    
    def read_group_evaluation(result):
        error_lists = []
        for fold in folds:
            fold_dir = result + fold
            error_lists.append(read_evaluation(fold_dir))   
        error_lists = np.array(error_lists)
        error_list = np.mean(error_lists, axis=0)
        return error_list
    
    error_dict = {}
    for result in results:
        if result == 'Multiple_gaze':
            error_list = []
            for scale in args.scales:
                error = read_group_evaluation(result + '/%.1f' % scale)
                error_list.append(error[0])
            error_dict[result] = error_list
        else:
            error_list = read_group_evaluation(result)
            error_dict[result] = error_list
    
    return error_dict


def fig4_plot_error(results, dataset, fig_name):
    error_dict = read_all_evaluations(results, dataset)

    color_set = {
        'Baseline': '#ff0a0a',
        'Baseline+FA': '#f2ce02',
        'Baseline+KD': '#ebff0a',
        'Baseline+FA+KD': '#209c05',
        'HR': '#000000'
    }

    # Setting
    plt.figure()
    plt.xlabel('Scale factor ' + r'$s$')
    plt.ylabel('Mean gaze error (degree)')
    plt.grid(visible=True, axis='y', lw=0.5)

    # Plot results
    for result in results:
        if result == 'HR':  # HR dashed line
            plt.axhline(error_dict['HR'][0], color=color_set[result], linestyle='--', label='HR image', lw=0.8)
        else:
            plt.plot(args.scales[1:], error_dict[result][1:], color=color_set[result], marker='o', markersize=3, linestyle='--', label=result)
            # Annotation
            if args.annotate:
                for i in range(len(error_dict[result][1:])):
                    plt.annotate('%.3f' % error_dict[result][i], xy=(args.scales[1:][i],error_dict[result][1:][i]), ha='center', size=5)
    
    # Save results
    #plt.title(DATASET_DICT[dataset]['title'])
    plt.legend(loc=2, fontsize='medium')
    output_path = '%s/%s.pdf' % (OUTPUT_DIR, fig_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def fig5_plot_error(results, labels, fig_name, dataset='mpii'):
    error_dict = read_all_evaluations(results, dataset)

    plotting_set = {
        'Baseline_color': ['#ffbaba', '#ff5252', '#a70000'],
        '+FA+KD_color': ['#a4fba6', '#30cb00', '#006203'],
        '+FA+KD': '#209c05',
        'markers': ['o', 'o', 'o'],
        'linestyle': ['dashed', 'dashed', 'dashed']
    }

    # Setting
    plt.figure()
    plt.xlabel('Scale factor ' + r'$s$')
    plt.ylabel('Mean gaze error (degree)')
    plt.grid(visible=True, axis='y', lw=0.5)

    # Plot results
    for i in range(len(results)):
        result = results[i]
        if '+FA+KD' in result:
            plt.plot(args.scales[1:], error_dict[result][1:], color=plotting_set['+FA+KD_color'][i//2],
                     marker=plotting_set['markers'][i//2], markersize=3, linestyle='--', label=labels[i])
        else:
            plt.plot(args.scales[1:], error_dict[result][1:], color=plotting_set['Baseline_color'][i//2],
                     marker=plotting_set['markers'][i//2], markersize=3, linestyle='--', label=labels[i])    

    # Save results
    #plt.title(DATASET_DICT[dataset]['title'])
    plt.legend(loc=2, fontsize='medium')
    output_path = '%s/%s.pdf' % (OUTPUT_DIR, fig_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def tab1_compare_to_baselines(results):
    utmv_error_dict = read_all_evaluations(results, 'utmv')
    mpii_error_dict = read_all_evaluations(results, 'mpii')
    
    # Models
    gaze_model = ArbGaze('resnet18', False, 0, None)
    ArbGaze_model = ArbGaze('resnet18', True, 4, None)
    ArbSR = MetaRDN()

    # For ArbSR models
    scales = np.arange(1.1, 4.1, 0.1)
    HR_img = np.ones((36,60))
    macs_ArbSR = []
    params_ArbSR = []
    for scale in scales:
        # Prepare inputs
        LR_img = cv2.resize(HR_img, dsize=None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
        LR_H, LR_W = LR_img.shape[0], LR_img.shape[1]
        LR_img = np.expand_dims(np.expand_dims(LR_img, axis=0), axis=0) # HW-> 11HW
        LR_img = torch.FloatTensor(LR_img)
        scale = torch.tensor(scale)
        # Count params and FLOPs
        scale_coord_map, _ = input_matrix_wpn(LR_H, LR_W, scale)
        macs, params = profile(ArbSR, inputs=(LR_img, scale_coord_map, scale))
        macs_ArbSR.append(macs)
        params_ArbSR.append(params)
    # Take mean over all scale factors
    macs_ArbSR = sum(macs_ArbSR) / float(len(macs_ArbSR))
    params_ArbSR = sum(params_ArbSR) / float(len(params_ArbSR))

    # Other models
    HR_img = np.expand_dims(np.expand_dims(HR_img, axis=0), axis=0)
    HR_img = torch.FloatTensor(HR_img)
    macs_gaze, params_gaze = profile(gaze_model, inputs=(HR_img, scale))    # Scale is not used
    macs_ArbGaze, params_ArbGaze = profile(ArbGaze_model, inputs=(HR_img, scale))    
    
    params_and_flops = []   # ArbSR+Gaze, MultipleGaze, ArbGaze
    params_and_flops.append([params_gaze+params_ArbSR, (macs_gaze+macs_ArbSR)*2])
    params_and_flops.append([params_gaze*30, macs_gaze*2]) # MAC = 2 * FLOP
    params_and_flops.append([params_ArbGaze, macs_ArbGaze*2])

    # Save as text file
    output_path = '%s/%s.txt' % (OUTPUT_DIR, 'tab1')
    write_file = open(output_path, 'w')
    write_file.write('\t\t\t\t\t\t\t\tutmv\t\t\tmpii\n')
    write_file.write('\t\t\tParams(M)\tFLOPs(G)\ts=2.0\ts=3.0\ts=4.0\ts=2.0\ts=3.0\ts=4.0\n')
    write_file.write('Arbitrary SR model\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (
        params_and_flops[0][0]/1e6, params_and_flops[0][1]/1e9,
        utmv_error_dict['ArbSR'][0], utmv_error_dict['ArbSR'][1], utmv_error_dict['ArbSR'][2],
        mpii_error_dict['ArbSR'][0], mpii_error_dict['ArbSR'][1], mpii_error_dict['ArbSR'][2]))
    write_file.write('multiple gaze model\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (
        params_and_flops[1][0]/1e6, params_and_flops[1][1]/1e9,
        utmv_error_dict['Multiple_gaze'][0], utmv_error_dict['Multiple_gaze'][1], utmv_error_dict['Multiple_gaze'][2],
        mpii_error_dict['Multiple_gaze'][0], mpii_error_dict['Multiple_gaze'][1], mpii_error_dict['Multiple_gaze'][2]))
    write_file.write('Proposed method\t\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (
        params_and_flops[2][0]/1e6, params_and_flops[2][1]/1e9,
        utmv_error_dict['Baseline+FA+KD'][0], utmv_error_dict['Baseline+FA+KD'][1], utmv_error_dict['Baseline+FA+KD'][2],
        mpii_error_dict['Baseline+FA+KD'][0], mpii_error_dict['Baseline+FA+KD'][1], mpii_error_dict['Baseline+FA+KD'][2]
        ))
    write_file.close()


def fig6_KD(results, labels, fig_name, dataset='mpii'):
    error_dict = read_all_evaluations(results, dataset)

    plotting_set = {
        'Baseline_color': ['#a70000', '#ff5252', '#ffbaba'],
        'KD_color': ['#a4fba6', '#30cb00', '#006203'],
        '+FA+KD': '#209c05',
        'markers': ['o', 'o', 'o'],
        'linestyle': ['dashed', 'dashed', 'dashed']
    }

    # Setting
    plt.figure()
    plt.xlabel('Scale factor ' + r'$s$')
    plt.ylabel('Mean gaze error (degree)')
    plt.ylim(5.0, 5.8)
    plt.grid(visible=True, axis='y', lw=0.5)

    # Plot results
    for i in range(len(results)):
        result = results[i]
        if 'HR' in result:
            plt.axhline(error_dict[result][0], color='k', linestyle='--', label=labels[i], lw=0.8)
        else:
            plt.plot(args.scales[1:], error_dict[result][1:], color=plotting_set['KD_color'][i],
                     marker=plotting_set['markers'][i//2], markersize=3, linestyle='--', label=labels[i])           

    # Save results
    #plt.title(DATASET_DICT[dataset]['title'])
    plt.legend(loc=2, fontsize='medium')
    output_path = '%s/%s.pdf' % (OUTPUT_DIR, fig_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def fig7_visualize_gaze_vector(dataset, fold, predefined_iterations):
    # Models
    baseline_model = ArbGaze('resnet18', False, 0, None)
    ArbGaze_model = ArbGaze('resnet18', True, 4, None)
    baseline_model.to(DEVICE)
    ArbGaze_model.to(DEVICE)

    # Get evaluation group(or id)
    _, TEST_SUBJECT_IDS = train_and_test_split(dataset, fold)

    # Path of trained models
    baseline_dir = '../results/%s/Baseline/f%d' % (dataset, fold)
    ArbGaze_dir = '../results/%s/Baseline+FA+KD/f%d' % (dataset, fold)

    # Baseline
    baseline_file = sorted(os.listdir(os.path.join(baseline_dir, 'saved_models')))[-1]
    baseline_file = os.path.join(baseline_dir, 'saved_models', baseline_file)
    baseline_state_dict = torch.load(baseline_file).copy()
    baseline_model.load_state_dict(baseline_state_dict['state_dict'])
    # ArbGaze
    ArbGaze_file = sorted(os.listdir(os.path.join(ArbGaze_dir, 'saved_models')))[-1]
    ArbGaze_file = os.path.join(ArbGaze_dir, 'saved_models', ArbGaze_file)
    ArbGaze_state_dict = torch.load(ArbGaze_file).copy()
    ArbGaze_model.load_state_dict(ArbGaze_state_dict['state_dict'])

    loss_function = GazeAngularLoss()
    loss_meter = AverageMeter()

    # Data
    data_path = '../datasets/%s.h5' % dataset
    test_dataset = GazeData(dataset, data_path=data_path, subject_ids=TEST_SUBJECT_IDS)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    baseline_model.eval()
    ArbGaze_model.eval()
    for i, (data_dict) in enumerate(loader, 1):
        if i > predefined_iterations[-1]:
            break
        if not (i in predefined_iterations):
            continue
        data_dict = send_data_dict_to_gpu(data_dict, DEVICE)
        HR_img = tensor_img_to_np_img((data_dict['HR_img'].cpu()+1)/2)[0]
        HR_H, HR_W = HR_img.shape[0], HR_img.shape[1] 

        # GT vector on HR image
        V_GT = data_dict['gaze'].cpu().numpy()
        PW_GT = vector_to_pitchyaw(-V_GT.reshape(1,3)).flatten()
        HR_vis = cv2.resize(HR_img, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        # Drawings
        center = (HR_vis.shape[1]//2, HR_vis.shape[0]//2)
        HR_thickness= 2
        BC_thickness= 1
        arrow_length = 40
        HR_vis = draw_gaze(HR_vis, center, PW_GT, arrow_length, HR_thickness, (255,0,0))

        loss_meter.reset()
        for scale in args.vis_scales:
            # Prepare inputs
            LR_img = cv2.resize(HR_img, dsize=None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
            BC_img = cv2.resize(LR_img, dsize=(HR_W,HR_H), interpolation=cv2.INTER_CUBIC)
            BC_input = np.expand_dims(np.expand_dims(BC_img, axis=0), axis=0)
            BC_input = torch.FloatTensor(BC_input).to(DEVICE)
            BC_input = 2 * BC_input / 255.0 - 1

            with torch.no_grad():
                V_BL, _ = baseline_model(BC_input, torch.tensor(scale).float())
                V_AG, _ = ArbGaze_model(BC_input, torch.tensor(scale).float())
            
            # Gaze differece on large scale factors
            # if scale >= 2.5:
            #     dif_gaze = loss_function(V_AG, V_BL)
            #     loss_meter.update(dif_gaze, 1)

            # Vector to Pitch-Yaw
            V_BL = V_BL.cpu().numpy()
            V_AG = V_AG.cpu().numpy()
            PW_BL = vector_to_pitchyaw(-V_BL.reshape(1,3)).flatten()
            PW_AG = vector_to_pitchyaw(-V_AG.reshape(1,3)).flatten()

            BC_vis = cv2.resize(BC_img, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            BC_vis = draw_gaze(BC_vis, center, PW_GT, arrow_length, HR_thickness, (255,0,0))
            BC_vis = draw_gaze(BC_vis, center, PW_BL, arrow_length, BC_thickness, (0,0,255))
            BC_vis = draw_gaze(BC_vis, center, PW_AG, arrow_length, BC_thickness, (0,255,0))
            HR_vis = np.append(HR_vis, BC_vis, axis=1)  # Append horizontally

        # Save visualizations when difference is larger than 5 degree
        # if loss_meter.avg > 5:
        output_path = '%s/visualization/%s' % (OUTPUT_DIR, dataset)
        prepare_path(output_path)
        output_path = '%s/%s_%d_%d.png' % (output_path, dataset, fold, i)
        cv2.imwrite(output_path, HR_vis)
            

if __name__ == '__main__':
    if 'fig4' in args.which:
        results = ['Baseline', 'Baseline+FA', 'Baseline+KD', 'Baseline+FA+KD', 'HR']
        fig4_plot_error(results, 'utmv', 'fig4_a')
        fig4_plot_error(results, 'mpii', 'fig4_b')

    if 'fig5' in args.which:
        results = ['Baseline/resnet10', 'Baseline+FA+KD/resnet10', 'Baseline', 'Baseline+FA+KD', 'Baseline/resnet34', 'Baseline+FA+KD/resnet34']
        labels = ['ResNet10', 'ResNet10+FA+KD', 'ResNet18', 'ResNet18+FA+KD', 'ResNet34', 'ResNet34+FA+KD']
        fig5_plot_error(results, labels, 'fig5_a')
        results = ['Baseline/vgg13_bn', 'Baseline+FA+KD/vgg13_bn', 'Baseline/vgg16_bn', 'Baseline+FA+KD/vgg16_bn', 'Baseline/vgg19_bn', 'Baseline+FA+KD/vgg19_bn']
        labels = ['VGG13', 'VGG13+FA+KD', 'VGG16', 'VGG16+FA+KD', 'VGG19', 'VGG19+FA+KD']
        fig5_plot_error(results, labels, 'fig5_b')

    if 'fig6' in args.which:
        results = ['Baseline+FA+KD/resnet10', 'Model_compression/R34to10', 'Baseline+FA+KD/resnet34']
        labels = ['ResNet10'+u'\u2192'+'ResNet10', 'ResNet34'+u'\u2192'+'ResNet10', 'ResNet34'+u'\u2192'+'ResNet34']
        fig6_KD(results, labels, 'fig6_a')
        results = ['Baseline+FA+KD/resnet10', 'Model_compression/R18to10', 'Baseline+FA+KD']
        labels = ['ResNet10'+u'\u2192'+'ResNet10', 'ResNet18'+u'\u2192'+'ResNet10', 'ResNet18'+u'\u2192'+'ResNet18']
        fig6_KD(results, labels, 'fig6_b')

    if 'fig7' in args.which:
        # Set scales to visualize
        if args.vis_scales is None:
            args.vis_scales = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        # Pre-defined image to be visualized
        predefined_dict = {
            'utmv':{
                # '1': [4382, 4878, 9229, 9253, 9341, 9773, 10326, 10374, 11230, 12878, 12998, 13334, 13342, 13375, 13599, 13855, 13878, 20910],
                # '2': [222, 318, 334, 558, 654, 670, 710, 974, 1062, 9305, 10225],
                # '3': [2856, 3328, 3344, 3480, 3689]
                '1': [3974, 13855],
                '2': [10025]
            },
            'mpii': {    
                # '4': [1933, 1963, 2031, 2055, 2090, 2182, 2206, 2218, 2330, 2373, 2429, 2471, 2755],
                # '5': [1203, 1881, 2176, 2373],
                # '6': [1772, 2370],
                # '7': [1116],
                # '9': [2009, 2254],
                # '10': [1529, 1588, 1614, 1691, 1716, 1730, 1740, 1761, 1844, 1845, 1868, 2111, 2213, 2276, 2386, 2645],
                # '11': [1317],
                # '12': [673, 2436, 2586],
                # '13': [1347],
                # '14': [441, 1624, 1777, 2254, 2467, 2698, 2858, 2917, 2933]
                '4': [2206],
                '5': [2176],
                '14': [1777]
            }
        }
        for dataset, item in predefined_dict.items():
            for fold, iterations in item.items():
                predefined_iterations = predefined_dict[dataset][fold]
                fig7_visualize_gaze_vector(dataset, int(fold), predefined_iterations)
    
    if 'tab1' in args.which:
        results = ['ArbSR', 'Multiple_gaze', 'Baseline+FA+KD']
        args.scales = [2.0, 3.0, 4.0]
        tab1_compare_to_baselines(results)