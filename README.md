Train locally on single GPU

```
bash single_machine.sh
```


Train on AWS on 8-machines
```
pip install -r requirements.txt
aws configure  (or set your AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_DEFAULT_REGION)
python launch.py --config=eight_machines
```


To reproduce the 21.544 perplexity on wt103 after 170 minutes of training on one instance
```
python launch.py --config=one_machine
```


