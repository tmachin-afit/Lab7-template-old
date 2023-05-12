import os
import re
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from tqdm.keras import TqdmCallback

fig_path = '/remote_home/EENG645a-Sp23/Lab6/Lab7-template/figures'
model_path = '/remote_home/EENG645a-Sp23/Lab6/Lab7-template/models'
log_path = '/remote_home/EENG645a-Sp23/Lab6/Lab7-template/logs'

def get_delta_s_delta_theta_from_xy(x: np.ndarray, y: np.ndarray,
                                    return_delta_theta=True, return_trig=False,
                                    back_fill_theta=False, back_fill_delta_d=False) -> np.ndarray:
    """Converts a sequence of x, y positions to relative distance between sequence steps.
    Can also return the relative angle between sequence steps if desired
    Can also return the sin and cos of the relative angle if desired.

    By default, the return sequence is the same length as the input sequence but the first time step is padded to be zero
    If desired the relative distance and/or angle can be backfilled (index 1 is copied to index 0)

    :param x: ndarray of the x positions. The actual data should be in the last dimension.
    :param y: ndarray of the y positions. The actual data should be in the last dimension.
    :param return_delta_theta: if true will return the relative angle
    :param return_trig: if true will return the sin and cos if the relative angle
    :param back_fill_theta: if true will make the first relative angle be equal to the second
    :param back_fill_delta_d: if true will make the first relative distance equal to the second
    :return: numpy ndarray with relative angle and distance between time steps.
    This will be same length as the input x,y.
    The actual data is stored in the last dimension in the order:
    [delta_d, delta_theta, sin(delta_theta), cos(delta_theta), theta)]
    The first values either set to zero or backfilled if backfill is set.
    :rtype: np.ndarray
    """
    reduce_to_2d = False
    if x.ndim != y.ndim:
        raise ValueError(f"x and y should have same dimensions not {x.ndim} and {y.ndim} respectively")
    # if the last dimension is 1, then strip it. For example turn shape of (5, 1000, 1) into (5, 1000)
    if x.shape[-1] == 1:
        x = x[..., 0]
        y = y[..., 0]
    if x.ndim < 2:
        x = x[None, ...]
        y = y[None, ...]
        reduce_to_2d = True

    delta_x = np.diff(x)
    delta_y = np.diff(y)

    # get absolute angle between each diff
    theta = np.concatenate([np.zeros(shape=(delta_y.shape[0], 1)), np.arctan2(delta_y, delta_x)], axis=-1)

    # back fill if necessary
    if back_fill_theta:
        theta[..., 0] = theta[..., 1]

    # get the relative distance and angle
    delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2)
    delta_theta = np.diff(theta)

    # get the trig values of the relative angles
    sin_delta_theta = np.sin(delta_theta)
    cos_delta_theta = np.cos(delta_theta)

    deltas = np.concatenate(
        [delta_s[..., None], delta_theta[..., None], sin_delta_theta[..., None], cos_delta_theta[..., None]], axis=-1)

    # pad the beginning with zeros to match original size
    zeros_shape = list(deltas.shape)
    zeros_shape[-2] = 1
    deltas = np.concatenate([np.zeros(zeros_shape), deltas], axis=-2)
    # the first cosine term should be 1 not zero
    deltas[..., 0, 3] = 1

    if back_fill_delta_d:
        deltas[..., 0, 0] = deltas[..., 1, 0]

    # only return what they asked for
    ret_columns = [0]
    if return_delta_theta:
        ret_columns += [1]
    if return_trig:
        ret_columns += [2, 3]
    ret = deltas[..., ret_columns]

    if reduce_to_2d:
        ret = ret[0]
    return ret


def get_xy_from_deltas(delta_d: np.ndarray,
                       delta_theta: np.ndarray,
                       initial_conditions: np.ndarray = None) -> np.ndarray:
    """Converts a sequence of relative distances and angle changes to 2D x,y positions
    Assumes the starting location is 0,0 and starting heading is 0 unless initial_conditions are given

    :param delta_d: relative distances between each step in the sequence
    :param delta_theta: relative angles between each step in the sequence
    :param initial_conditions: the starting x,y,theta point(s)
    :return: a numpy ndarray with the x,y positions in the last dimension
    :rtype: np.ndarray
    """
    reduce_to_2d = False
    if delta_d.ndim != delta_theta.ndim:
        raise ValueError(
            f"delta_d and delta_theta should have same dimensions not {delta_d.ndim} and {delta_theta.ndim} respectively")
    # if the last dimension is 1, then strip it. For example turn shape of (1000, 1) into (1000,)
    if delta_theta.shape[-1] == 1:
        delta_theta = delta_theta[..., 0]
        delta_d = delta_d[..., 0]
    if delta_theta.ndim < 2:
        delta_theta = delta_theta[None, ...]
        delta_d = delta_d[None, ...]
        reduce_to_2d = True
    if initial_conditions is None:
        initial_conditions = np.zeros(shape=(delta_d.shape[0], 3))

    theta = np.concatenate([np.zeros(shape=(delta_theta.shape[0], 1)), np.cumsum(delta_theta, axis=-1)], axis=-1)
    theta += initial_conditions[..., 2:3]

    delta_x = delta_d * np.cos(theta[..., :-1] + delta_theta)
    delta_y = delta_d * np.sin(theta[..., :-1] + delta_theta)

    x = np.cumsum(delta_x, axis=-1) + initial_conditions[..., 0:1]
    y = np.cumsum(delta_y, axis=-1) + initial_conditions[..., 1:2]

    position = np.concatenate([x[..., None], y[..., None]], axis=-1)

    if reduce_to_2d:
        position = position[0]
    return position


def read_dataframes(file_root: str,
                    vicon_column_names: typing.List[str],
                    imu_column_names: typing.List[str]
                    ) -> typing.Tuple[typing.List[pd.DataFrame],
                                      typing.List[pd.DataFrame],
                                      typing.List[pd.DataFrame],
                                      typing.List[pd.DataFrame]]:
    """Reads the csv files of the Oxford Inertial Tracking Dataset into lists of DataFrames
    breaks into the training and test sets

    :param file_root: the root where the Oxford data is with the name "Oxford Inertial Tracking Dataset"
    :param vicon_column_names: the column names to use for the vicon data
    :param imu_column_names: the column names to use for the imu data
    :return: a tuple with four lists. Each list has one dataframes per data collect.
            The first two lists are the training input and labels. The last two lists are the test inputs and labels.
    """
    # read in raw data
    ignore_first: int = 2000
    vi_list_train: typing.List[pd.DataFrame] = []
    imu_list_train: typing.List[pd.DataFrame] = []
    vi_list_test: typing.List[pd.DataFrame] = []
    imu_list_test: typing.List[pd.DataFrame] = []
    for root, dirs, files in os.walk(file_root, topdown=False):
        if 'handheld' in root and 'syn' in root:
            for i in range(len(files) // 2):
                vi_name = os.path.join(root, f"vi{i + 1}.csv")
                imu_name = os.path.join(root, f"imu{i + 1}.csv")
                if os.path.exists(vi_name) and os.path.exists(imu_name):
                    vi_temp = pd.read_csv(vi_name, names=vicon_column_names)[ignore_first:]
                    imu_temp = pd.read_csv(imu_name, names=imu_column_names)[ignore_first:]

                    deltas = get_delta_s_delta_theta_from_xy(vi_temp['translation.x'].values,
                                                             vi_temp['translation.y'].values)
                    vi_temp['delta_s'] = deltas[..., 0]
                    vi_temp['delta_theta'] = deltas[..., 1]

                    data_num = int(re.search(r"data(?P<num>\d+)", root).group("num"))
                    if data_num < 5:
                        vi_list_train.append(vi_temp)
                        imu_list_train.append(imu_temp)
                    else:
                        vi_list_test.append(vi_temp)
                        imu_list_test.append(imu_temp)
    print(f"Got {len(vi_list_train)} data frames")

    return vi_list_train, imu_list_train, vi_list_test, imu_list_test


def get_dataset_from_lists(imu_list: typing.List[pd.DataFrame],
                           vi_list: typing.List[pd.DataFrame],
                           input_columns=None,
                           output_columns=None,
                           seq_len: int = 10,
                           batch_size: int = 32) -> tf.data.Dataset:
    """Take the raw sequences adn break them up into smaller sequences for training
    the dataframes in the lists will be 2D (total_sequence_timestep, data)
    The output of the dataset should be 3D (sample, window_timestep, data)

    :param imu_list: the list of pandas dataframes for the imu data
    :param vi_list: the list of pandas dataframes for the vi data
    :param input_columns: the columns to use as input
    :param output_columns: the columns to use as output
    :param seq_len: the number of time steps in each output sequence (how many timesteps in each window)
    :param batch_size: the size of batches to use for training
    :return: a dataset object that will return batches of data to train on
    """
    ds: tf.data.Dataset
    return ds


def build_model(seq_len: int,
                input_data_size: int,
                output_data_size: int,
                batch_size: int = None,
                stateful: bool = False) -> Model:
    model: Model
    return model


def main():
    fileRoot = os.path.join("/opt", "data", "Oxford_Inertial_Tracking_Dataset")

    # print our column names that we got from the top level description
    vicon_column_names: typing.List[str] = "Time Header translation.x translation.y translation.z " \
                                           "rotation.x rotation.y rotation.z rotation.w".split(' ')
    print(vicon_column_names)
    imu_column_names: typing.List[str] = "Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) " \
                                         "rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) " \
                                         "gravity_x(G) gravity_y(G) gravity_z(G) " \
                                         "user_acc_x(G) user_acc_y(G) user_acc_z(G) " \
                                         "magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)".split(
        ' ')
    print(imu_column_names)

    # make our variables for the experiment
    input_columns: typing.List[str] = []
    output_columns: typing.List[str] = []
    model_name: str = "model.h5"
    num_samples: int
    seq_len: int
    use_dataset: bool

    vi_list_train, imu_list_train, vi_list_test, imu_list_test = read_dataframes(file_root=file_root,
                                                                                 vicon_column_names=vicon_column_names,
                                                                                 imu_column_names=imu_column_names)

    # turn the raw data into sequences
    ds_train = get_dataset_from_lists(imu_list=imu_list_train,
                                      vi_list=vi_list_train,
                                      input_columns=input_columns,
                                      output_columns=output_columns,
                                      seq_len=seq_len,
                                      batch_size=batch_size)

    ds_valid = get_dataset_from_lists(imu_list=imu_list_valid,
                                      vi_list=vi_list_valid,
                                      input_columns=input_columns,
                                      output_columns=output_columns,
                                      seq_len=seq_len,
                                      batch_size=batch_size)

    input_data_size: int
    output_data_size: int

    # do any other pre-processing
    if not os.path.exists(model_name):
        # build and save model
        model: Model = build_model(seq_len,
                                   input_data_size,
                                   output_data_size,
                                   stateful=False)
        # now fit the model
        model.fit()

        model.save(model_name)
    else:
        model = load_model(model_name)

    # build a model to predict for very long sequences for our visualization
    # we will most likely need to call the predict function multiple times since our sequence will not fit into GPU RAM
    # if we call the predict function multiple times we need to make the model stateful to pass the state to the next predict
    # remember to set the `stateful` property of each of the RNN layers in the build model function
    model_stateful: Model = build_model(seq_len,
                                        input_data_size,
                                        output_data_size,
                                        stateful=True)

    # make our test data (use training/validation at first) and then create predictions from it
    # remember at this point we are working with longer sequences
    # models need to be able to predict for as many timesteps as required and thus predicting on one timestep at a time works well
    x_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray

    # Visualization

    # print the whole values
    plt.figure()
    # plot our true delta_s values

    # plot our predicted delta_s values

    # label axes
    plt.title('delta_s Whole Values')
    plt.xlabel('timestep')
    plt.ylabel('distance (m)')
    plt.legend()

    # print the top down view
    plt.figure()
    # plot the true position

    # plot our predicted position

    # label axes
    plt.title('Top Down View')
    plt.xlabel('vicon x axis (m)')
    plt.ylabel('vicon y axis (m)')
    plt.legend()


if __name__ == "__main__":
    main()
