# UTMultiview dataset
for f in $(seq 1 1 3);
do
    python main.py --dataset_name utmv --fold $f --save_path ../results/utmv/teacherf${f} \
                   --train_img_type HR --test_img_type HR
done
# MPIIGaze dataset
for f in $(seq 0 1 14);
do
    python main.py --dataset_name mpii --fold $f --save_path ../results/mpii/teacherf${f} \
                   --train_img_type HR --test_img_type HR
done
