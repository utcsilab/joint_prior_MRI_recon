# bin/bash
# write bash script looping over different samples and running inference_joint.py
for sample in {0..1000}
do
for R in {8,4,6,10}
do
    python inference_joint.py --gpu 0 --sample $sample --conditioning joint --R $R --ACS_perc 0.04 --num_steps 500 --sigma_max 10 --l_ss 1.0
done
done
