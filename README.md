# MonoVits

reference
https://github.com/zxcqlf/MonoViT

For training, please download monodepth2, replace the depth network, and revise the setting of the depth network, the optimizer and learning rate according to trainer.py.

Because of the different torch version between MonoViT and Monodepth2, the func transforms.ColorJitter.get_params in dataloader should also be revised to transforms.ColorJitter.

## Setting
### 공통 설치 항목
- 실험 환경 : Ubuntu 18.04 CUDA 9.1
  
```bash
$ conda create -n monovit python=3.7 anaconda
$ pip3 install torch==1.8.0 torchvision==0.9.0 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
$ pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
$ pip3 install mmsegmentation==0.11.0
$ pip3 install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
$ pip install timm einops IPython
```

## Dataset
- 데이터셋 구조
  - 폴더명과 파일 이름을 다음과 같이 수정해주었음
    ```bash
    |-data/
    |  |--infra/
    |  |  |--s_1/
    |  |  |  |--0000000000.jpg
    |  |  |--s_2/
    |  |  |  |--0000000000.jpg
    |  |  |...
    |  |  |--s_198/
    |  |  |  |--0000000000.jpg
    |  |--rgb/
    |  |  |--s_1/
    |  |  |  |--0000000000.jpg
    |  |  |--s_2/
    |  |  |  |--0000000000.jpg
    |  |  |...
    |  |  |--s_125/
    |  |  |  |--0000000000.jpg


## Training

- Download the ImageNet-1K pretrained MPViT model(https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth) to ./ckpt/.

- By default models and tensorboard event files are saved to ./tmp/<model_name>. This can be changed with the --log_dir flag.

- data_path에는 dataset의 경로를 넣어주고 dataset에는 train에 사용할 데이터셋의 종류를 넣어준다.
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 14 --num_epochs 20 --model_name {train_model_name} --data_path=../data --dataset=infra --learning_rate 5e-5
```


- custom dataset을 train에 사용하는 방법
  1. create own dataset class inheriting from datasets/mono_dataset.py
  2. build your own dataset, and create a new split file
  3. add your dataset class in init.py and trainer.py(datasets_dict)

