for dataset in fake imdb
do
    for lr in 0.01 0.001 ## 0. 0.01 0.1 0.5 1.0 2.0 3 5 10 15 20
    do
      for weight_decay in 0 0.1 0.3 0.01 0.001 0.0001
        do
        for mask_ratio in 0.05 0.1 0.15 0.2 0.25 0.3
        do
        for weight in 0.001 0.01 0.1 0.3 0.5 0.7 1 ###2 5 10 15
        do
           python main_ssl.py --weight_decay $weight_decay --lr $lr --weight $weight --mask_ratio $mask_ratio &
        done
        wait
        done
        wait
        done
        wait
        done
        wait
done