import warnings

# seaborn uses deprecated pandas methods internally which raise warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
