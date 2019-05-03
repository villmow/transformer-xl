Train locally on single GPU

```
bash single_machine.sh
```


Train small network on AWS on 8-machines
```
pip install -r requirements.txt
aws configure
python launch.py --config=eight_machines
```

Reproduce 21.5 perplexity on wikitext-103 after 2 hours 50 minutes and one machine
```
python launch.py --config=one_machine
```

# Throughput results

To reproduce "44x faster" than original number

90 seconds per epoch on 8-machines vs 3800 seconds originally
```
python launch.py --config=eight_machines
```


# To reproduce throughput numbers reported 

```
python launch.py --config=one_machine_large
python launch.py --config=two_machines_large
python launch.py --config=four_machines_large
python launch.py --config=eight_machines_large
python launch.py --config=sixteen_machines_large

# update log directories in the file below
python generate_throughput_numbers.py
```