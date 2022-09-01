'''
The code below was written with reference to:
https://github.com/hysts/pytorch_mpiigaze/blob/master/tools/preprocess_mpiigaze.py
https://github.com/swook/faze_preprocess/blob/5c33caaa1bc271a8d6aad21837e334108f293683/create_hdf_files_for_faze.py
'''


import os, cv2, csv, h5py
import numpy as np
import pandas as pd
from scipy import io


OUTPUT_DIR = '../datasets'
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def draw_gaze(image_in, init_point, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    image_out = image_in
    if len(image_in.shape) == 2:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    # Draw arrow
    cv2.arrowedLine(image_out, tuple(np.round(init_point).astype(np.int32)),
                    tuple(np.round([init_point[0] + dx, init_point[1] + dy]).astype(int)),
                    color, thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


def preprocess_mpii(mpii_dict):
    def get_eval_info(subject_id, annot_path):
        annot_path = os.path.join(annot_path, '%s.txt' % subject_id)
        df = pd.read_csv(annot_path, delimiter=' ', header=None, names=['path', 'side'])
        df['day'] = df.path.apply(lambda path: path.split('/')[0])
        df['filename'] = df.path.apply(lambda path: path.split('/')[1])
        df = df.drop(['path'], axis=1)
        return df
    
    output_path = os.path.join(OUTPUT_DIR, mpii_dict['output_file'])
    input_path = os.path.join(mpii_dict['data_path'], 'Data', 'Normalized')
    annot_path = os.path.join(mpii_dict['data_path'], 'Evaluation Subset', 'sample list for eye image')
    
    subjects = sorted(os.listdir(input_path))
    for subject in subjects:
        print('Processing %s....' % subject)

        to_write = {}
        def add(key, value):
            if key not in to_write:
                to_write[key] = value
            else:
                to_write[key].append(value)
        
        left_images = dict()
        left_poses = dict()
        left_gazes = dict()
        right_images = dict()
        right_poses = dict()
        right_gazes = dict()
        file_names = dict()
        
        # Read all data
        subject_path = os.path.join(input_path, subject)
        for mat_file in sorted(os.listdir(subject_path)):
            mat_data = io.loadmat(os.path.join(subject_path, mat_file), struct_as_record=False, squeeze_me=True)

            data = mat_data['data']
            day = mat_file[:5]
        
            left_images[day] = data.left.image
            left_poses[day] = data.left.pose
            left_gazes[day] = data.left.gaze

            right_images[day] = data.right.image
            right_poses[day] = data.right.pose
            right_gazes[day] = data.right.gaze

            file_names[day] = mat_data['filenames']

            if not isinstance(file_names[day], np.ndarray):
                left_images[day] = np.array([left_images[day]])
                left_poses[day] = np.array([left_poses[day]])
                left_gazes[day] = np.array([left_gazes[day]])
                right_images[day] = np.array([right_images[day]])
                right_poses[day] = np.array([right_poses[day]])
                right_gazes[day] = np.array([right_gazes[day]])
                file_names[day] = np.array([file_names[day]])
        
        # Read evaluation subset info
        df = get_eval_info(subject, annot_path)
        images = []
        poses = []
        gazes = []
        for _, row in df.iterrows():
            day = row.day
            index = np.where(file_names[day] == row.filename)[0][0]
            if row.side == 'left':
                image = left_images[day][index]
                pose = left_poses[day][index]
                gaze = left_gazes[day][index]
            else:
                image = right_images[day][index]#[:, ::-1]
                pose = right_poses[day][index]
                gaze = right_gazes[day][index]
            
            images.append(image)
            poses.append(pose)
            gazes.append(gaze)
        images = np.asarray(images).astype(np.uint8)
        poses = np.asarray(poses).astype(np.float32)
        gazes = np.asarray(gazes).astype(np.float32)

        # Add data to be written
        add('images', images)
        add('gazes', gazes)

        for key, values in to_write.items():
            to_write[key] = np.asarray(values)
            print('%s: ' % key, to_write[key].shape)
        
        # Visualization
        for i in range(len(to_write['images'])):
            if i % 30 == 0:
                img = to_write['images'][i]
                to_visualize = img.copy()
                pw = vector_to_pitchyaw(-to_write['gazes'][i].reshape(3,1).T).flatten()
                ow, oh = img.shape[1], img.shape[0]
                to_visualize = draw_gaze(to_visualize, (0.5*ow,0.5*oh), pw, length=30, thickness=2)
                cv2.imshow('img', to_visualize)
                key = cv2.waitKey(10)
                if key == 27:
                    break
                elif key == 83 or key == 115:
                    save_path = OUTPUT_DIR + '/%s_%d.png' % (subject, i)
                    cv2.imwrite(save_path, to_visualize)
                else:
                    continue
        
        # Write to HDF
        with h5py.File(output_path, 'a' if os.path.isfile(output_path) else 'w') as f:
            if subject in f:
                del f[subject]
            group = f.create_group(subject)
            for key, values in to_write.items():
                group.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )
        print('')


def preprocess_utmv(utmv_dict):
    NORMALIZED_CAMERA = {
        'focal_length': 960,
        'distance': 600,
        'size': (60, 36),
    }
    NORM_CAMERA_MATRIX = np.array(
        [
            [NORMALIZED_CAMERA['focal_length'], 0, 0.5*NORMALIZED_CAMERA['size'][0]],
            [0, NORMALIZED_CAMERA['focal_length'], 0.5*NORMALIZED_CAMERA['size'][1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    def read_monitor_pose(path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip(' [').rstrip(']\n').split()
        # Translation
        pose_t = [float(lines[1][i]) for i in range(len(lines[1]))]
        pose_t = np.array(pose_t)
        # Rotation
        pose_R = []
        for i in range(2, 5):
            pose_R.append([float(lines[i][j]) for j in range(len(lines[i]))])
        pose_R = np.array(pose_R)
        return pose_t, pose_R

    def read_gt_2d(path):
        gt_2d = []
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            img_num = -1
            for row in csv_reader:
                if img_num == -1:
                    img_num += 1
                    continue
                gt_2d.append([float(row[1]), float(row[2])])
        gt_2d = np.array(gt_2d)
        return gt_2d
    
    def read_cparams(path):
        cparams = []
        cparam_files = sorted(os.listdir(path))
        for cparam_file in cparam_files:
            f = open(os.path.join(path, cparam_file))
            lines = f.readlines()
            f.close()
            # Camera parameters
            cparam = []
            for i in range(1, 4):
                lines[i] = lines[i].split()
                cparam.append([float(lines[i][j]) for j in range(len(lines[i]))])
            cparams.append(cparam)
        cparams = np.array(cparams)
        return cparams
    
    def read_images(path):
        imgs = []
        img_files = sorted(os.listdir(path))
        for img_file in img_files:
            imgs.append(cv2.imread(os.path.join(path, img_file)))
        return imgs
    
    def read_head_pose_and_landmarks(path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip(' [').rstrip(']\n').split()
        # Translation
        pose_t = [float(lines[1][i]) for i in range(len(lines[1]))]
        pose_t = np.array(pose_t)
        # Rotation
        pose_R = []
        for i in range(2, 5):
            pose_R.append([float(lines[i][j]) for j in range(len(lines[i]))])
        pose_R = np.array(pose_R)
        # Landmarks
        ldmks = []
        for i in range(6, 12):
            ldmks.append([float(lines[i][j]) for j in range(len(lines[i]))])
        ldmks = np.array(ldmks)
        return pose_t, pose_R, ldmks
    
    output_path = os.path.join(OUTPUT_DIR, utmv_dict['output_file'])
    input_path = utmv_dict['data_path']

    subjects = sorted(os.listdir(input_path))
    for subject in subjects:
        print('Processing %s....' % subject)

        to_write = {}
        def add(key, value):
            if key not in to_write:
                to_write[key] = [value]
            else:
                to_write[key].append(value)

        subject_path = os.path.join(input_path, subject, 'raw')
        # Read 2d target & monitor pose -> world target 3d
        monitor_pose_t, monitor_pose_R = read_monitor_pose(os.path.join(subject_path, 'monitor.txt'))
        monitor_targets_2d = read_gt_2d(os.path.join(subject_path, 'gazedata.csv'))  # 160x2
        monitor_targets_3d = np.concatenate((monitor_targets_2d, np.zeros((160,1))), axis=1)  # Concat 0 for z-axis
        world_targets_3d = np.matmul(monitor_pose_R, monitor_targets_3d.T).T + np.expand_dims(monitor_pose_t, axis=0)
        world_targets_3d = np.concatenate((world_targets_3d, np.ones((160,1))), axis=1)   # Homogeneous

        # 160 gaze directions
        gaze_directions = sorted(os.listdir(subject_path))[1:-1]
        for i in range(len(gaze_directions)):
            gaze_direction_path = os.path.join(subject_path, gaze_directions[i])

            # 8 camera parameters & 8 images
            cparams = read_cparams(os.path.join(gaze_direction_path, 'cparams'))
            imgs = read_images(os.path.join(gaze_direction_path, 'images'))
            world_P = cparams[0]    # Camera0 = world

            world_target_3d = world_targets_3d[i]
            world_head_pose_t, world_head_pose_R, world_ldmks_3d = read_head_pose_and_landmarks(os.path.join(gaze_direction_path, 'headpose.txt'))
            world_ldmks_3d = np.concatenate((world_ldmks_3d, np.ones((6,1))), axis=1)   # Homogeneous

            # Each camera coord.
            for j in range(len(imgs)):
                img = imgs[j]   # Image
                cparam = cparams[j]

                K = world_P[:, :3]
                P = cparam
                R_t = np.matmul(np.linalg.inv(K), P)
                R = R_t[:, :3]
                t = R_t[:, 3]
                head_pose_R = np.dot(R, world_head_pose_R)

                target_3d = np.dot(R_t, world_target_3d.T).reshape(3, 1)
                ldmks_3d = np.dot(R_t, world_ldmks_3d.T).T

                # Each eye
                r_g_o = np.mean(ldmks_3d[0:2], axis=0).reshape(3, 1)    # Right gaze origin
                l_g_o = np.mean(ldmks_3d[2:4], axis=0).reshape(3, 1)
                rl_g_o = [r_g_o, l_g_o]

                norm_imgs = np.zeros((NORMALIZED_CAMERA['size'][1], NORMALIZED_CAMERA['size'][0] * 2), dtype=np.uint8)    # Right & left
                norm_gazes = np.zeros((2,3))
                for k in range(len(rl_g_o)):
                    g_o = rl_g_o[k]
                    gaze = target_3d - g_o
                    gaze = gaze / np.linalg.norm(gaze)

                    distance = np.linalg.norm(g_o)
                    z_scale = NORMALIZED_CAMERA['distance'] / distance
                    S = np.eye(3, dtype=np.float64)
                    S[2, 2] = z_scale

                    hRx = head_pose_R[:, 0]
                    forward = (g_o / distance).reshape(3)
                    down = np.cross(forward, hRx)
                    down /= np.linalg.norm(down)
                    right = np.cross(down, forward)
                    right /= np.linalg.norm(right)
                    R = np.c_[right, down, forward].T   # Rotation matrix R

                    W = np.dot(np.dot(NORM_CAMERA_MATRIX, S),
                               np.dot(R, np.linalg.inv(K)))
                    
                    ow, oh = NORMALIZED_CAMERA['size']
                    norm_img = cv2.warpPerspective(img, W, (ow,oh))
                    norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
                    norm_img = cv2.equalizeHist(norm_img)

                    # Gaze in normalized camera
                    norm_gaze = np.dot(R, gaze)
                    norm_gaze = norm_gaze / np.linalg.norm(norm_gaze)

                    norm_gazes[k] = norm_gaze.reshape(1, 3)
                    norm_imgs[:, ow*k:ow*k+ow] = norm_img
                
                add('images', norm_imgs)
                add('gazes', norm_gazes.astype(np.float32))

                # Visualization
                if (8 * i + j) % 30 == 0:
                    to_visualize = norm_imgs.copy()
                    ow, oh = norm_imgs.shape[1], norm_imgs.shape[0]
                    # Right
                    pw = vector_to_pitchyaw(-norm_gazes[0].reshape(1,3)).flatten()
                    to_visualize = draw_gaze(to_visualize, (0.25*ow,0.5*oh), pw, length=30, thickness=2)
                    # Left
                    pw = vector_to_pitchyaw(-norm_gazes[1].reshape(1,3)).flatten()
                    to_visualize = draw_gaze(to_visualize, (0.75*ow,0.5*oh), pw, length=30, thickness=2)
                    cv2.imshow('img', to_visualize)
                    key = cv2.waitKey(10)
                    if key == 27:
                        break
                    elif key == 83 or key == 115:
                        save_path = OUTPUT_DIR + '/%s_%d.png' % (subject, 8*i+j)
                        cv2.imwrite(save_path, to_visualize)
                    else:
                        continue

        for key, values in to_write.items():
            to_write[key] = np.asarray(values)
            print('%s: ' % key, to_write[key].shape)
        
        # Write to HDF
        with h5py.File(output_path, 'a' if os.path.isfile(output_path) else 'w') as f:
            if subject in f:
                del f[subject]
            group = f.create_group(subject)
            for key, values in to_write.items():
                group.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )
        print('')


if __name__ == '__main__':
    mpii_dict = {
        'data_path': '/home/hgk/hgk/DB/gaze/MPIIGaze',
        'output_file': 'mpii.h5'
    }
    utmv_dict = {
        'data_path': '/home/hgk/hgk/DB/gaze/UTMultiview',
        'output_file': 'utmv.h5'
    }

    preprocess_mpii(mpii_dict)
    preprocess_utmv(utmv_dict)