set -x

# Or generate your own using https://github.com/attardi/wikiextractor
#aws s3 cp s3://yaroslavvb2/data/wikiextracted.tar .
wget https://s3.amazonaws.com/yaroslavvb2/data/wikiextracted.tar
tar -xf wikiextracted.tar
# Flatten all the files.
find wikiextracted/ -iname 'wiki*' -type f -exec sh -c 'jq -r .text {} > {}.txt' \;
# Make moses-ified version for eval.
find . -name wiki_00.txt -o -name wiki_01.txt -type f -exec sh -c 'mkdir -p $(dirname ../data/wikitext-big/{}) && sacremoses
 tokenize -l en < {} > ../data/wikitext-big/{}' \;
# Preprocess all the data.
source activate pytorch_p36
for i in {1..$(nproc)}
do
    python data_utils.py --datadir=wikiextracted/ --dataset=wiki &
done

wait
echo 'all done'
