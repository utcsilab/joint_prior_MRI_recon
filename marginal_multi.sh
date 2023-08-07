# bin/bash
for sample in {0..1000}
do
for R in {8,4,6,10}
do
    python inference_marginal.py --gpu 2 --sample $sample --contrast_recon PD --R $R --ACS_perc 0.04 --num_steps 500  --sigma_max 10 --l_ss 1.0
    python inference_marginal.py --gpu 2 --sample $sample --contrast_recon PDFS --R $R --ACS_perc 0.04 --num_steps 500  --sigma_max 10 --l_ss 1.0

done

done