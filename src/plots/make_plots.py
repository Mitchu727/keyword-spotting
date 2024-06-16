from src.utils import get_plots_directory
import os
import pandas as pd

training_loss_directory = get_plots_directory() / "training_loss"
validation_accuracy_directory = get_plots_directory() / "validation_accuracy"
validation_loss_directory = get_plots_directory() / "validation_loss"
output_directory = get_plots_directory() / "output"

dataset_analysis_directory = get_plots_directory() / "dataset_analysis"
for dirname in os.listdir(dataset_analysis_directory):
    mfcc = pd.read_csv(dataset_analysis_directory / dirname / "mfcc.csv", index_col=None, header=None)
    mfcc = mfcc.sort_values(by=0)
    mfcc.plot.bar(x=0, y=1, legend=False, xlabel="").get_figure().savefig(output_directory / f"{dirname}_mfcc")
    classes = pd.read_csv(dataset_analysis_directory / dirname / "classes.csv", index_col=None, header=None)
    classes = classes.sort_values(by=0)
    classes.plot.bar(x=0, y=1, legend=False, xlabel="").get_figure().savefig(output_directory / f"{dirname}_classes")


# for filename in os.listdir(validation_loss_directory):

# training_loss_df = pd.DataFrame(index=range(0, 1000))
# for filename in os.listdir(training_loss_directory):
#     df = pd.read_csv(training_loss_directory / filename)
#     training_loss_df[filename[:-4]] = list(df['Value'])
#
#
# plt = training_loss_df.plot(figsize=[10, 6])
# plt.get_figure().savefig(output_directory / "training_loss")
#
# validation_accuracy_df = pd.DataFrame(index=range(0, 1000))
# for filename in os.listdir(validation_accuracy_directory):
#     df = pd.read_csv(validation_accuracy_directory / filename)
#     validation_accuracy_df[filename[:-4]] = list(df['Value'])
#
#
# plt = validation_accuracy_df.plot(figsize=[10, 6])
# plt.get_figure().savefig(output_directory / "validation_accuracy")
#
# validation_loss_df = pd.DataFrame(index=range(0, 1000))
# for filename in os.listdir(validation_loss_directory):
#     df = pd.read_csv(validation_loss_directory / filename)
#     validation_loss_df[filename[:-4]] = list(df['Value'])
#
#
# plt = validation_loss_df.plot(figsize=[10, 6])
# plt.get_figure().savefig(output_directory / "validation_loss")