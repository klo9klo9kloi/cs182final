# cs182final

## CS182 RL Project

Reinforcement learning algorithms have been successfully applied to various single-task settings such as games and automotive control, but generalize-ability to new tasks and environments remains an ongoing area of exploration. In this work we use OpenAI's ProcGen testing suite to evaluate the effect of the number of training levels on the generalize-ability of vanilla DQN to unseen levels of easy Fruitbot. We also attempt to explore the effects of Prioritized Experience Replay and ICM on generalize-ability. Overall, we find that increasing the number of training levels as well as adding augmentations to DQN both aid with generalize-ability.

## Requirements

Running the train and test process require specific versions of the program. To install the required versions, follow the command line signature below

```
pip install -r requirements.txt
```

## Running the Model

The model can be run on any of the procgen environments that support 'easy' mode, with the option to specify the arguments. Follow the command line signature below

```
python train_dqn.py ENV_NAME [--num_steps <NUM_STEPS>] [--num_levels <NUM_LEVELS>] [--pr] [--replay_buffer_size <NUM_SIZE_BUFFER>] [--seed <SEED>]
```

## Testing the Model

The training process automatically saves the model in the target directory. Follow the command line signature below to test the model against some number of randomly generated seeds. Generated JSON results will be located in the same directory as the model.

```
python test_dqn.py ENV_NAME [--logdir <MODEL_LOCATION>] [--run_test_num <NUM_TESTS>]
```
