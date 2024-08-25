# bin/bash
for seed in {5,6,7,8,9}
do
for sample in {0..100}
do
for R in {8,4,6,10}
do
    python inference_marginal.py --gpu 0 --sample $sample --contrast_recon PD --R $R --num_steps 500  --sigma_max 10 --l_ss 1.0 --seed $seed --task SuperRes

done
done
done