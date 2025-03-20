#The .py version of the Jupyter Notebook used to clean up the data for processing

import pandas as pd

from google.colab import drive

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/MarchMadness.csv')

df.info()

df.head()

"""Only care about March Madness Teams"""

df.drop(df.loc[df['Post-Season Tournament']!='March Madness'].index, inplace=True)

df.info()

"""Cool, now the file isnt the size of a neutron star.
Prior to 2007 height stats werent recorded so anything prior will be dropped, as will 2020 due to covid making the year an anomoly.


"""

df.drop(df.loc[df['Season']<=2007].index, inplace=True)

df.drop(df.loc[df['Season']==2020].index, inplace=True)

df.info()

"""Other oddities remain with incomplete data, Im just gonna drop the columns which seem to be missing significant data across the remaining years. Non Steal Turnovers would be a measure of carelessness in a team which is reflected in other places."""

df = df.drop(['Region', 'Top 12 in AP Top 25 During Week 6?', 'Active Coaching Length','DFP', 'NSTRate', 'RankNSTRate', 'OppNSTRate', 'RankOppNSTRate', 'Short Conference Name', 'Mapped Conference Name', 'Current Coach', 'Full Team Name', 'Since'], axis=1)

df.info()

column_to_move = df.pop("Mapped ESPN Team Name")

# insert column with insert(location, column_name, column_value)

df.insert(0, "Mapped ESPN Team Name", column_to_move)

df.head()

df = df.sort_values(['Season'], ascending=False)

df.to_csv('drive/MyDrive/Data.csv', index=False)

df.drop(df.loc[df['Season']!=2025].index, inplace=True)

df.to_csv('drive/MyDrive/2025Teams.csv', index=False)

"""Ive taken the original file of 9000+ rows and compressed it into two separate lighter & more relevant data files. The remainder of the data manipulation will occur in the python files."""