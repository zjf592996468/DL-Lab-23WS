# Team14

- Jifeng Zhou (st181366)
- Yang Cao (st186635)

# How to run the code

## Attention!

The checkpoint logic of our code is different from the original code.
We are, if there is a checkpoint it is loaded.

So please rename the `ckpt` file before running it if you don't want to use checkpoint.
Equally, if you want to use a good checkpoint just drag the file into `experiments`
and rename it to '**ckpt**', for P2 is '**har-ckpt**'.

## P1: Diabetic Retinopathy Detection

Train default '**cnn**' model with **L2-regularization** on IDRID dataset for binary classification:

Run batch file: `sbatch run.sh`

Train other model:

* vgg(need raise train steps to 12e4):

  `python3 main.py --train=True --multi_class=False --classification=False --model='vgg'
   --l2_loss=flase`

* Tune hyperparameters:

  `python3 main.py --train=True --multi_class=False --classification=False --model='cnn'
   --l2_loss=True`

* Deep visualization(need ckpt in `experiments`):

    `python3 visual.py`

* Transfer learning:

    `python3 main.py --train=True --multi_class=False --classification=False --model='effnet'
      --l2_loss=flase`


Train '**cnn**' model for multi-class classification:

* Use classification:

  `python3 main.py --train=True --multi_class=True --classification=True --model='cnn'
   --l2_loss=True`

* Use regression:

  `python3 main.py --train=True --multi_class=True --classification=False --model='cnn'
   --l2_loss=True`
* Use evaluation(use the corresponding model):\
   `python3 main.py --train=False...`

## P2: Human Activity Recognition

Train RNN model with '**Bidirectional LSTM**' layer on HAPT dataset:

    `python3 main.py --train=True`

Train RNN model with other RNN layer:

* LSTM:

  `python3 main.py -- train=True --layer='LSTM'`

* Bidirectional GRU:

  `python3 main.py --train=True --layer='Bidirectional GRU'`

* GRU:\
  `python3 main.py --train=True --layer='GRU'`

* Evaluation:\
`python3 main.py --train=False`

# Results

## P1: Diabetic Retinopathy Detection

### Binary Model Test Accuracy

| Model                 | CNN | VGG | EfficientNet* |
|-----------------------|-----|-----|---------------|
| **Test Accuracy (%)** | 89  | 74  | 88            |

### Multi-Classification Model Test Accuracy

| Model                 | CNN-Classification | CNN-Regression |
|-----------------------|--------------------|----------------|
| **Test Accuracy (%)** | 57                 | 43             |

*_The best record of EfficientNet was achieved by
splitting the first 20% of the original Train set into the Validation set._\
*_The results of this project vary greatly because the test set is small.
We took the highest record and uploaded the ckpt file to result._


## P2: Human Activity Recognition

| Model                     | RNN with bidirectional LSTM |
|---------------------------|-----------------------------|
| **Test Accuracy (%)**     | 95                          |
| **Balanced Accuracy (%)** | 83                          |

Our model achieved great success in basic activities, but need improvement when it comes to
postural transition.

