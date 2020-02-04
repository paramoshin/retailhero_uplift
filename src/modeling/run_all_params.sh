echo base: &&
python train.py --refit True &&
echo \nlevel_1: &&
python train.py --level_1 True --refit True &&
echo \nreceny: &&
python train.py --recency True --refit True &&
echo \nfrequency: &&
python train.py --frequency True --refit True &&
echo \nrecency, frequency: &&
python train.py --frequency True --recency True --refit True &&
echo \nlevel_1, recency, frequency: &&
python train.py --level_1 True --recency True --frequency True --refit True

