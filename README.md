# Keiki

​    A bullet hell game Platform for research purpose (especially danmaku generation) written in Python.

​    In this platform, you can design "danmakus" by implement Danmaku class provided in the platform. The platform guarantee subclasses of this class can be encoded into a parameter sequence for training Generative Adversarial Nets (GANs). We will update the documentations for the danmaku designing APIs at https://github.com/PneuC/Keiki/wiki/API-Docs soon.

### Eviorment we have tested:

​    Python 3.7.6

​    Pygame 2.0.1

​    Pytorch 1.5.0

​    Numpy 1.18.1

​    Seaborn 0.11.1

​    Matlotlib 3.3.1

### How to use:

#### Encoding Danmaku Class into Parametric Sequence:

​    Run run_make.py to encode all the subclasses of *Danmaku* found in *data/code* folder

#### Run Game Demo:

​    The recomanded ways is put encoded danmakus (in *npy* format) into *danmakus* folder, then run *run_game.py* to start. The platform will load all the npy files in this folder as the danmakus of the boss. You can also import danmaku classes at logic/boss.py and then add the imported classes into Boss.spells attributes.

##### How to play:

​    Use **direction keys** to move. 

​    Hold **Z** to keep shooting

​    Hold **Shift** to keep slow mode. In slow mode the your moving speed will decrease to enhance operating accuracy.

​    You can also press **S** to skip the current danmaku.

**Training GANs:**

​    We provide 3 implemented GANs in *generator* folder. For each GANs you can run *train.py* at the corresonding folder to train it. You can execute *python train.py -h* or *python train.py --help* to check the parameters for the training. 