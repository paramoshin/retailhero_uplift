# echo base: &&
# python train.py --refit True &&
# echo $'\nlevel_1:' &&
# python train.py --level_1 True --refit True &&
# echo $'\nreceny:' &&
# python train.py --recency True --refit True &&
# echo $'\nfrequency:' &&
# python train.py --frequency True --refit True &&
# echo $'\nrecency, frequency:' &&
# python train.py --frequency True --recency True --refit True &&
# echo $'\nlevel_1, recency:' &&
# python train.py --level_1 True --recency True --refit True &&
# echo $'\nlevel_1, frequency:' &&
# python train.py --level_1 True --frequency True --refit True &&
# echo $'\nlevel_1, recency, frequency': &&
# python train.py --level_1 True --recency True --frequency True --refit True
echo $'\nlda:' &&
python train.py --lda True --refit True
echo $'\nw2v:' &&
python train.py --w2v True --refit True
echo $'\nlda, w2v:' &&
python train.py --lda True --w2v True --refit True
echo $'\nlda, level_1:' &&
python train.py --lda True --level_1 True --refit True
echo $'\nw2v, level_1:' &&
python train.py --w2v True --level_1 True --refit True
echo $'\nlda, w2v, level_1:' &&
python train.py --lda True --w2v True, --level_1 True --refit True
