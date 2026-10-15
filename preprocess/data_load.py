import os

import dill


def load_preprocessed_data(filepath):
    data = dill.load(open(filepath, 'rb'))
    print(f"Data loaded from {filepath}")
    return data


def preprocess_data(args, task=None):
    if task is None:
        task = args.task

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base, f'data/mimic3/processed_data/{task}/processed_data.pkl')

    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_data_path}\n"
            f"The pre-built processed_data.pkl for task '{task}' is required."
        )

    return load_preprocessed_data(processed_data_path)
