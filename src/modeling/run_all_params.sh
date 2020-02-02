echo level_1: &&
python train.py --level_1 True --refit True &&
echo receny: &&
python train.py --recency True --refit True &&
echo frequency: &&
python train.py --frequency True --refit True &&
echo recency, frequency: &&
python train.py --frequency True --recency True --refit True &&
echo level_1, recency, frequency: &&
python train.py --level_1 True --recency True --frequency True --refit True

