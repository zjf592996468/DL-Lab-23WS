# Team14

- Jifeng Zhou (st181366)
- Yang Cao (st186635)

# How to run the code

## Attention!

The checkpoint mechanism in our code differs from the original implementation. In our case, if a checkpoint exists, it
is automatically loaded. So it's important to note that the code cannot be run consecutively due to the need to prevent
the automatic loading of checkpoint files.

Therefore, if you prefer not to utilize a checkpoint, please rename the checkpoint file before execution. Conversely, if
you wish to use a specific, high-quality checkpoint, simply drag the file into the '**experiments**' directory and
rename it to '**ckpt**'. For project 2, rename it to '**har-ckpt**'.

## P1: Diabetic Retinopathy Detection

- Train '**cnn**' model with **L2-regularization** on IDRID dataset for **binary classification**:

  Run batch file: `sbatch run.sh`

- Train other model:

    * vgg(need to raise train steps to **12e4**):

      `python3 main.py --train=True --multi_class=False --classification=True --model='vgg' --l2_loss=False`

- Tune hyperparameters:

  Run batch file: `sbatch tune.sh`

- Deep visualization (need ckpt in `experiments`):

  `python3 visual.py`

- Transfer learning:

    * EfficientNet(**4e4** may be better):

      `python3 main.py --train=True --multi_class=False --classification=True --model='effnet' --l2_loss=False`

- Train '**cnn**' model for multi-class classification:

    * Regression:

      `python3 main.py --train=True --multi_class=True --classification=False --model='cnn' --l2_loss=True`

    * Classification(**6e4** may be better):

      `python3 main.py --train=True --multi_class=True --classification=True --model='cnn' --l2_loss=True`

- Evaluation (use the corresponding model):

  `python3 main.py --train=False ...`

## P2: Human Activity Recognition

- Train '**myRNN**' model with '**Bidirectional LSTM**' layer on HAPT dataset:

  Run batch file: `sbatch run.sh`

- Train model with other RNN layer:

    * LSTM:

      `python3 main.py -- train=True --layer='LSTM'`

    * Bidirectional GRU:

      `python3 main.py --train=True --layer='Bidirectional GRU'`

    * GRU:

      `python3 main.py --train=True --layer='GRU'`

- Evaluation:

  `python3 main.py --train=False`

# Results

## P1: Diabetic Retinopathy Detection

### Binary Model Test Accuracy

| Model                 | CNN  | VGG  | EfficientNet* |
|-----------------------|------|------|---------------|
| **Test Accuracy (%)** | 89.3 | 74.1 | 88.3          |

### Multi-Classification Model Test Accuracy

| Model                 | CNN-Classification | CNN-Regression |
|-----------------------|--------------------|----------------|
| **Test Accuracy (%)** | 56.2               | 45.6           |

*_The best record of EfficientNet was achieved by
splitting the first 20% of the original Train set into the Validation set._\
*_The results of this project vary greatly because the test set is small.
We took the highest record and uploaded the ckpt file to result._

## P2: Human Activity Recognition

| Model                     | RNN with bidirectional LSTM |
|---------------------------|-----------------------------|
| **Test Accuracy (%)**     | 95.5                        |
| **Balanced Accuracy (%)** | 85.0                        |

Our model achieved great success in basic activities, but need improvement when it comes to
postural transition.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
