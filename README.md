# MeetingSummarization

## How to add:
- `git clone https://github.com/wxj77/MeetingSummarization.git`

## How to compile:
-  `pip install -r requirements.txt`
-  `python setup.py install`

## How to commit:
- `git add .`
- `git commit -m 'I did what'`
- `git push origin branch_name`

## How to request merge:
...


## How to run HMNet:
- build docker image `sudo docker build . -t hmnet`
- clone HMNet into MeetingSummarization: `git clone https://github.com/microsoft/HMNet`
- download `model.pt` into `HMNet/ExampleInitModel/HMNet-pretrained`
- run docker with gpu and attach volume: `sudo docker run --gpus all -v /home/wji/language/MeetingSummarization:/MeetingSummarization -it hmnet /bin/bash`
- test nvidia: `nvidia-smi`
- test cuda: `python -c 'import torch; print(torch.cuda.is_available());'`
- go into HMNet directory: cd /MeetingSummarization/HMNet
- run train script with 1 gpu: `CUDA_VISIBLE_DEVICES="0" mpirun -np 1 --allow-run-as-root python PyLearn.py train ExampleConf/conf_hmnet_AMI`
