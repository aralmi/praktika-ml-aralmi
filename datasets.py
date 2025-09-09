import pandas as pd
import numpy as np
from scipy.signal import decimate
import os
import random
import requests
from rarfile import RarFile
import re
import warnings
from scipy.io import loadmat

class CurrentsVoltagesVibrationsDataset:
    """
    A class for loading the Netherlands EM and Pump Currents, Voltages and Vibrations dataset from paper: https://www.sciencedirect.com/science/article/pii/S235234092301017X
    """
    def __init__(self, dataset_path: str, states_list: list) -> None:
        """
        Initializes a CurrentsVoltagesVibrationsDataset object.

        Args:
            dataset_path (str): The path to the dataset folder which contains Electrical state folder and Vibration state folder.
            states_list (list): A list of tuples containing the name of the state, electrical state folder path, vibration state folder path and percentage of measurements to be used for testing.
        """
        self.path = dataset_path
        self.states = states_list

    def load_data(self) -> dict:
        """
        A method for loading the Netherlands EM and Pump Currents, Voltages and Vibrations dataset from paper: https://www.sciencedirect.com/science/article/pii/S235234092301017X


        Returns:
            dict: A dictionary containing train and test data for each state with electrical and vibration data.
        """
        data = {}

        # Iterate over each state
        for state in self.states:
            print(f'{state[0]} state is loading...')
            # Iterate over each type of data (electrical and vibration)
            for path in [state[1], state[2]]:
                if path is None:
                    continue
                columns_name = []
                train_list = []
                test_list = []
                label, train_test_split = state[-2:]

                # Get the file names of the data
                files = os.listdir(os.path.join(self.path, path))

                # Get the number of measurements in the data
                measurement_num = pd.read_csv(os.path.join(self.path, path, files[0])).drop(columns='time').shape[1]
                
                # There was an error in the data for the healthy motor state at speed 1, so we remove one measurement
                if 'Electric_Motor-2_100_time-healthy 1' in files[0]:
                    print(f'Found {state[0]} at speed 100, removing measurement number 14 with error in it...')
                    measurement_num -= 1
                
                # Get the column names of the measurements
                measurement_cols = [str(i) for i in range(measurement_num)]

                # Randomly select the columns to be used for test
                test_cols = random.sample(measurement_cols, int(train_test_split * measurement_num))
                print(f'train cols = {[x for x in measurement_cols if x not in test_cols]}, test cols = {test_cols}')
                # Iterate over each file in the data
                for ch_file in files:
                    # Read the file into a DataFrame
                    channel = pd.read_csv(os.path.join(self.path, path, ch_file)).drop(columns='time')
                    
                    # There was an error in the data for the healthy motor state at speed 1, so we remove one measurement
                    if 'Electric_Motor-2_100_time-healthy 1' in ch_file:
                        channel = channel.drop(columns='14')
                        channel.columns = np.arange(len(channel.columns)).astype(str)
                    # Get the channel number from the file name
                    ch_num = int(ch_file[-5:-4])

                    # Add the channel number to the list of column names
                    if 'Vibr' in path:
                        columns_name.append(f'vibr_{ch_num}')
                    else:
                        columns_name.append(f'i_{ch_num}' if ch_num < 4 else f'v_{ch_num-3}')

                    # Add the flattened measurement data to the lists
                    train_df = channel.drop(columns=test_cols)
                    train_list.append(train_df.to_numpy().astype(np.float32).flatten(order='F')) # order='F' - columnwise flattening
                    test_df = channel[test_cols]
                    test_list.append(test_df.to_numpy().astype(np.float32).flatten(order='F')) # order='F' - columnwise flattening

                # Create DataFrames from the lists of data
                train_data = pd.DataFrame(np.vstack(train_list).T, columns=columns_name)
                test_data = pd.DataFrame(np.vstack(test_list).T, columns=columns_name)

                # Create the name for the state
                name = state[0] + ('_vibr' if 'Vibr' in path else '')

                # Add the data to the dictionary
                data[name] = {
                    'train_data': train_data,
                    'test_data': test_data
                }
        return data
                    
    @staticmethod
    def _apply_park_transform(data: pd.DataFrame, get_phase: bool=True) -> pd.DataFrame:
        """
        Applies the Park transform to a given DataFrame.

        The Park transform is a way to transform 3-phase electrical data into a 2-phase signal, which adds more information.

        Args:
            data (pd.DataFrame): A DataFrame containing the 3-phase electrical data.

        Returns:
            pd.DataFrame: The DataFrame with the added 2-phase electrical data.
        """
        # Calculate the alpha, beta components and the instantaneous amplitude of the current and voltage (if present)
        data['i_alpha'] = (2 * data['i_1'] - data['i_2'] - data['i_3']) / 3
        data['i_beta'] = (data['i_2'] - data['i_3']) / np.sqrt(3)
        data['instantaneous_i_amplitude'] = np.sqrt(data['i_alpha'] ** 2 + data['i_beta'] ** 2)
        if get_phase:
            data['instantaneous_i_phase'] = np.arctan2(data['i_beta'], data['i_alpha'])

        if 'v_1' in data.columns:
            data['v_alpha'] = (2 * data['v_1'] - data['v_2'] - data['v_3']) / 3
            data['v_beta'] = (data['v_2'] - data['v_3']) / np.sqrt(3)
            data['instantaneous_v_amplitude'] = np.sqrt(data['v_alpha'] ** 2 + data['v_beta'] ** 2)
            if get_phase:
                data['instantaneous_v_phase'] = np.arctan2(data['v_beta'], data['v_alpha'])

        return data

    @staticmethod
    def apply_park_transform(data: dict, get_phase: bool=True) -> dict:
        """
        Applies the Park transform to the electrical data in the given dataset.

        Args:
            data (dict): A dictionary containing train and test pd.DataFrames for each state with electrical and vibration data.
            get_phase (bool, optional): Whether to get the phase of the current and voltage. Defaults to True.

        Returns:
            dict: A dictionary containing train and test data for each state with added park-transformed data.
        """
        for state_name, state_data in data.items():
            if '_vibr' in state_name:
                continue

            for name, df in state_data.items():
                data[state_name][name] = CurrentsVoltagesVibrationsDataset._apply_park_transform(df, get_phase=get_phase)

        return data


    @staticmethod
    def reshape_data(self, data: dict, window_size_seconds: float, sampling_rate: int=20000) -> dict:
        """
        Reshapes the data in the given dataset to the specified time window size.

        The data is reshaped to (n_windows, window_size, n_channels) and the window size is
        calculated by multiplying the sampling frequency by the window size in seconds.

        Args:
            data (dict): A dictionary containing train and test data for each state.
            window_size_seconds (float): The time window size in seconds.

        Returns:
            dict: A dictionary containing train and test data for each state with reshaped data.
        """
        # Calculate the number of samples in the window
        samples_per_window = int(sampling_rate * window_size_seconds)

        # Check that the window size is a multiple of the main signal period
        main_signal_period_seconds = 1 / 50
        assert float.is_integer(window_size_seconds / main_signal_period_seconds), \
            f'Window size {window_size_seconds} seconds should be a multiple of the main signal period {main_signal_period_seconds} seconds to divide evenly'

        # Reshape the data for each state
        for state_name, state_datasets in data.items():
            for name, df in state_datasets.items():
                # Reshape the data to (n_windows, window_size, n_channels)
                if type(df) is pd.DataFrame:
                    data[state_name][name] = df.values.reshape(-1, samples_per_window, df.shape[-1])
                else:
                    data[state_name][name] = df.reshape(-1, samples_per_window, df.shape[-1])

        return data
    
    @staticmethod
    def downsample_data(data: dict, downsample_factor: int) -> dict:
        """
        Downsamples the data in the given dataset by a given factor.

        Args:
            data (dict): A dictionary containing train and test data for each state.
            downsample_factor (int): The factor by which to downsample the data.

        Returns:
            dict: A dictionary containing train and test data for each state with downsampled data.
        """
        # Downsample the data for each state
        for state_name, state_datasets in data.items():
            for name, df in state_datasets.items():
                # Reshape data to (n_windows, initial_signal_length_samples, n_channels)
                data[state_name][name] = df.values.reshape(-1, 20000 * 15, df.shape[-1])
                # Downsample the data by a given factor
                data[state_name][name] = decimate(data[state_name][name], q=downsample_factor, axis=1)
                # Reshape data back to (N, n_channels)
                data[state_name][name] = pd.DataFrame(data[state_name][name].reshape(-1, data[state_name][name].shape[-1]), columns=df.columns)
        return data
    
    
class PaderbornDataset:
    def __init__(self, path: str, download: bool=True, url: str='https://groups.uni-paderborn.de/kat/BearingDataCenter/') -> None:
        self.path = os.path.normpath(path)
        self.url = url
        self.bearings = {
            'Healthy': [
                'K001', # Run-in Period >50 h, Radial load 1000-3000 N, Speed 1500-2000 rpm
                'K002', # Run-in Period 19 h, Radial load 3000 N, Speed 2900 rpm
                'K003', # Run-in Period 1 h, Radial load 3000 N, Speed 3000 rpm
                'K004', # Run-in Period 5 h, Radial load 3000 N, Speed 3000 rpm
                'K005', # Run-in Period 10 h, Radial load 3000 N, Speed 3000 rpm
                'K006', # Run-in Period 16 h, Radial load 3000 N, Speed 2900 rpm
            ],
            'Artificial': {
                'OR': [
                    'KA01', # Damage level 1, EDM
                    'KA03', # Damage level 2, engraver
                    'KA05', # Damage level 1, engraver
                    'KA06', # Damage level 2, engraver
                    'KA07', # Damage level 1, drilling
                    'KA08', # Damage level 2, drilling
                    'KA09', # Damage level 2, drilling
                ],
                'IR': [
                    'KI01', # Damage level 1, EDM 
                    'KI03', # Damage level 1, engraver
                    'KI05', # Damage level 1, engraver
                    'KI07', # Damage level 2, engraver
                    'KI08', # Damage level 2, engraver
                ]
            },
            'Real': {
                'OR': [
                    'KA04', # Damage level 1, fatigue: pitting, single point
                    'KA15', # Damage level 1, plastic deform.: indentations, single point
                    'KA16', # Damage level 2, fatigue: pitting, repetitive damage with random arrangement within single point
                    'KA22', # Damage level 1, fatigue: pitting, single point
                    'KA30', # Damage level 1, plastic deform.: indentations, repetitive damage with random arrangement, distributed
                ],
                'IR': [
                    'KI04', # Damage level 1, fatigue: pitting, multiple damage without repetition, single point
                    'KI14', # Damage level 1, fatigue: pitting, multiple damage without repetition, single point
                    'KI16', # Damage level 3, fatigue: pitting, single point
                    'KI17', # Damage level 1, fatigue: pitting, repetitive damage with random arrangement, single point
                    'KI18', # Damage level 2, fatigue: pitting, single point
                    'KI21', # Damage level 1, fatigue: pitting, single point
                ],
                'Both': [
                    'KB23', # Damage level 2, fatigue: pitting, multiple damage with random arrangement within single point
                    'KB24', # Damage level 3, fatigue: pitting, multiple damage without repetition, distributed
                    'KB27', # Damage level 1, plastic deform.: indentations, multiple damage with random arrangement, distributed
                ]
            }
        }
        
        if download:
            self._download_dataset()
          
    def _download_dataset(self) -> None:
        """
        Downloads the bearings dataset from the given url and extracts the rar files in the appropriate folders.
        
        The folders are created in the following structure:
        - Healthy
            - bearing1
            - bearing2
        - Artificial
            - OR
                - bearing1
                - bearing2
            - IR
                - bearing1
                - bearing2
            - Both
                - bearing1
                - bearing2
        - Real
            - OR
                - bearing1
                - bearing2
            - IR
                - bearing1
                - bearing2
            - Both
                - bearing1
                - bearing2
        """
        
        for damage_method in self.bearings:
            method_path = os.path.join(self.path, damage_method)
            if not os.path.exists(method_path):
                os.makedirs(os.path.join(self.path, damage_method), exist_ok=True)
            
            if damage_method != 'Healthy':
                for damage_type in self.bearings[damage_method]:
                    cur_path = os.path.join(method_path, damage_type)
                    if not os.path.exists(cur_path):
                        os.mkdir(cur_path)
                        
                    for bearing in self.bearings[damage_method][damage_type]:
                        bearing_path = os.path.join(cur_path, bearing)
                        if not os.path.exists(bearing_path):
                            file_url = self.url + bearing + '.rar'
                            self._download_extract_rar(file_url, cur_path)
            else:
                cur_path = method_path 
                for bearing in self.bearings[damage_method]:
                    bearing_path = os.path.join(cur_path, bearing)
                    if not os.path.exists(bearing_path):
                        file_url = self.url + bearing + '.rar'
                        self._download_extract_rar(file_url, cur_path)

    def load_data(self, fault_methods, fault_types, rpm, torque, radial_force, only_currents=True):
        """
        Loads the data from the given fault methods, fault types, rpm, torque and radial force.
        
        Parameters
        ----------
        fault_methods : list
            List of fault methods to load data from.
        fault_types : list
            List of fault types to load data from.
        rpm : int
            The rpm of the motor.
        torque : float
            The torque of the motor.
        radial_force : int
            The radial force of the motor.
        only_currents : bool, optional
            Whether to only load the phase currents data. Defaults to True.
        
        Returns
        -------
        data : dict
            A dictionary with the loaded data, where each key is a column and the value is a numpy array.
        """
        data = {}
        file_filter = f'N{rpm//100:02d}_M{int(torque*10):02d}_F{radial_force//100:02d}_'
        
        for fault_num, fault in enumerate(fault_types):
            for method in fault_methods:
                method_path = os.path.join(self.path, method)
                if fault == 'Healthy':
                    cur_path = os.path.join(self.path, fault)
                else:
                    cur_path = os.path.join(method_path, fault)
                
                for walk_path, dirs, files in os.walk(cur_path):
                    dirs.sort() # inplace directories sorting to get bearing folders in right order
                    if len(files):
                        # Filtering by rpm, torque and radial force
                        files = [f for f in files if file_filter in f]
                        if len(files):
                            # Sorting by file number
                            files.sort(key=lambda x: int(x.split('_')[-1][:-4]))
                            # Loading .mat files
                            for measurement_num, file in enumerate(files):
                                data_mat = loadmat(os.path.join(walk_path, file))[file[:-4]][0][0]
                                
                                # Adding data as measurement
                                # for frame in data_mat[1][0]:
                                #     col_name = str(frame[4][0])
                                #     if col_name not in data:
                                #         data[col_name] = []
                                #     data[col_name].append(resize_signal(frame[2][0]))

                                for frame in data_mat[2][0]:
                                    col_name = str(frame[0][0])
                                    if only_currents and col_name not in ['phase_current_1', 'phase_current_2']:
                                        continue
                                    if col_name not in data:
                                        data[col_name] = []
                                    data[col_name].append(self.resize_signal(frame[2][0]))
                                
                                if 'measurement' not in data:
                                    data['measurement'] = []
                                if 'class' not in data:
                                    data['class'] = []
                                if 'bearing_name' not in data:
                                    data['bearing_name'] = []
                                data['measurement'].append(np.ones(256000, dtype=int) * measurement_num)
                                data['class'].append(np.ones(256000, dtype=int) * fault_num)
                                data['bearing_name'].append([walk_path.split('/')[-1]] * 256000)
                                # if len(fault_methods):
                                #     if 'fault_method' not in data:
                                #         data['fault_method'] = []
                                #     data['fault_method'].append([method] * 256000)
                        else:
                            raise Exception(f'No files found for parameters: rpm={rpm}, torque={torque}, radial_force={radial_force}')
        for key in list(data.keys()):
            data[key] = np.concatenate(data[key])
        return data
    
    @staticmethod    
    def resize_signal(data: np.ndarray, target_size: int=256000, threshold: int=20000, possible_lenghts: list=[5, 16000, 256000]) -> np.ndarray:
        """
        Resizes a given signal to a target size by either padding or repeating the signal.

        Args:
            data (np.ndarray): The signal to be resized.
            target_size (int): The desired length of the resized signal.
            threshold (int): The minimum length of the signal. If the signal is shorter than this threshold, it is padded with the last element.
            possible_lenghts (list): A list of possible lengths of the signal. The signal is resized to the closest length in this list.

        Returns:
            np.ndarray: The resized signal.
        """
        curr_len = len(data)
        closest_len_idx = np.argmin(np.abs(np.array(possible_lenghts) - curr_len))
        closest_len = possible_lenghts[closest_len_idx]
        
        if curr_len >= closest_len:
            # If the signal is longer than the closest length, truncate it to the closest length
            data = data[:closest_len]
        else:
            # If the signal is shorter than the closest length, pad it with the last element
            pad_size = closest_len - curr_len
            data = np.concatenate([data, np.full(pad_size, data[-1], dtype=data.dtype)])
        
        if closest_len != target_size:
            # If the signal is not the target size, repeat it to the target size
            data = np.repeat(data, target_size // closest_len)    
            
        return data
                        
    @staticmethod
    def _download_extract_rar(url: str, dirpath: str, max_retries: int = 3) -> None:
        """
        Downloads a RAR file from a given URL and extracts its contents to a specified directory.

        Args:
            url (str): The URL of the RAR file to be downloaded.
            dirpath (str): The directory path where the file should be extracted.

        Raises:
            Exception: If the filename cannot be found in the URL headers.

        Warns:
            UserWarning: If the download fails due to an unsuccessful HTTP response.

        The function performs the following steps:
        1. Sends a GET request to the provided URL to download the file.
        2. Extracts the filename from the 'content-disposition' header of the response.
        3. Saves the downloaded content to the specified directory.
        4. Extracts the contents of the RAR file.
        5. Removes the downloaded RAR file after extraction.
        """

        import time
    
        for attempt in range(max_retries):
            try:
                print(f"Попытка {attempt + 1} загрузки {url}")
                
                # Более надежная загрузка
                response = requests.get(url, allow_redirects=True, stream=True, timeout=30)
                response.raise_for_status()
                
                # Проверка размера файла
                expected_size = int(response.headers.get('content-length', 0))
                if expected_size == 0:
                    print("Предупреждение: Сервер не сообщил размер файла")
                
                filename = url.split('/')[-1]
                filepath = os.path.join(dirpath, filename)
                
                # Загрузка по частям с проверкой
                downloaded_size = 0
                with open(filepath, 'wb') as archive:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            archive.write(chunk)
                            downloaded_size += len(chunk)
                
                print(f"Загружено {downloaded_size} байт")
                
                # Проверка размера
                if expected_size > 0 and downloaded_size != expected_size:
                    print(f"Размер не совпадает: ожидалось {expected_size}, получено {downloaded_size}")
                    os.remove(filepath)
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise Exception("Не удалось загрузить файл полностью")
                
                # Проверка, что файл действительно является RAR
                try:
                    with RarFile(filepath, 'r') as test_archive:
                        test_archive.infolist()  # Попытка прочитать содержимое
                    print("Файл RAR валиден")
                except Exception as e:
                    print(f"Файл RAR поврежден: {e}")
                    os.remove(filepath)
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise
                
                # Извлечение архива
                with RarFile(filepath, 'r') as archive:
                    archive.extractall(dirpath)
                
                # Удаление архива после успешного извлечения
                os.remove(filepath)
                print(f"Успешно извлечен: {filename}")
                return
                
            except Exception as e:
                print(f"Ошибка при попытке {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Ждем перед следующей попыткой
                else:
                    raise
