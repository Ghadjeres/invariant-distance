import os

import pandas as pd
import seaborn as sns
from deepPermutations.data_utils import PACKAGE_DIR
from matplotlib import pyplot as plt


def plot_results_directory():
    results_dir = os.path.join(PACKAGE_DIR,
                               'results')
    model_types_and_files = os.listdir(results_dir)
    for model_dir_rel in model_types_and_files:
        model_dir_abs = os.path.join(results_dir,
                                     model_dir_rel)
        if not os.path.isdir(model_dir_abs):
            continue

        model_names_and_files = os.listdir(results_dir)
        for model_name_rel in model_names_and_files:
            model_name_abs = os.path.join(model_dir_abs,
                                          model_name_rel)
            if not os.path.isdir(model_name_abs):
                continue

            csv_files = filter(lambda s: s.endswith('.csv'),
                               os.listdir(model_name_abs))
            for csv in csv_files:
                csv_filename = os.path.join(model_name_abs,
                                            csv)

                base_filename = os.path.splitext(csv)[0]
                fig_filepath = os.path.join(model_name_abs,
                                            f'{base_filename}.svg'
                                            )
                plot_csv(csv_filename,
                         fig_filepath=fig_filepath)


def plot_csv(csv_filename, fig_filepath):
    hist_data = []
    df = pd.read_csv(csv_filename,
                     sep=', ',
                     header=0)
    for label in df['label'].unique():
        hist_data.append(df[df['label'] == label]['distance'])
    sns.set()
    fig, ax = plt.subplots()
    for a in hist_data:
        sns.distplot(a, ax=ax,
                     kde=True)
    # sns.plt.show()
    sns.plt.savefig(os.path.join(fig_filepath),
                    format='svg')


if __name__ == '__main__':
    plot_results_directory()
