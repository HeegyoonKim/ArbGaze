# UTMultiview dataset
for f in $(seq 1 1 3);
do
    python main.py --dataset_name utmv --fold $f --save_path ../results/utmv/Baseline+FA+KD/f${f} \
                   --load_from_teacher --FA_module --KD_loss_type MSE
done
# MPIIGaze dataset
for f in $(seq 0 1 14);
do
    python main.py --dataset_name mpii --fold $f --save_path ../results/mpii/Baseline+FA+KD/f${f} \
                   --load_from_teacher --FA_module --KD_loss_type MSE
done
