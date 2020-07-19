import numpy as np
import pandas as pd
from enum import Enum

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class KindNormalization(Enum):
	Scaling = 1,
	Zscore = 2,

class TimeSeriesData(Dataset):
	def __init__(self, input_path, n_columns, idx_class, normalise, kind_normalization):

		np.set_printoptions(suppress=True)

		# com index_col ja não inclui a coluna index 
		dataframe = pd.read_csv(input_path, header=0, index_col=['Date'], engine='python')

		#self.x_scaler = MinMaxScaler(feature_range=(0, 1))
		data = []

		if kind_normalization == KindNormalization.Zscore:
			self.x_scaler = StandardScaler() # mean and standart desviation
			self.y_scaler = StandardScaler() # mean and standart desviation

		elif kind_normalization== KindNormalization.Scaling:
			self.x_scaler = MinMaxScaler(feature_range=(0, 1))
			self.y_scaler = MinMaxScaler(feature_range=(0, 1))

		#i_split = round(len(data) * split)
		#print("[Data] Splitting data at %d with %s" %(i_split, split))

		# iloc é uma propiedade que funciona so com Dataframe e não se apliquei df.values

		x_data = dataframe.iloc[0:,0:n_columns]
		y_data = dataframe.iloc[0:,0:idx_class]
		y_data = y_data.shift(-1, axis=0)

		frame = [x_data, y_data]
		result = pd.concat(frame, axis=1)
		result = result.dropna()

		self.x_data = result.iloc[0:,0:n_columns]
		self.y_data = result.iloc[0:,-idx_class:] # the last column
		
		self.len = len(self.x_data) 

		if normalise:
			self.x_data = self.x_scaler.fit_transform(self.x_data)
			self.y_data = self.y_scaler.fit_transform(self.y_data)
			self.x_data = pd.DataFrame(self.x_data)
			self.y_data = pd.DataFrame(self.y_data)
		#else:
			#data = pd.DataFrame(dataframe.values)

		print("[Data] shape data X: ", self.x_data.shape)
		print("[Data] shape data y: ", self.y_data.shape)
		print('[Data] len:', self.len)

		# no pode retornar
		#return (x_data.values, y_data.values)
	
	def load_data_series(self, split=0):

		#self.data = self.data.values
		i_split = round(len(self.x_data) * split)

		print("[Data] Splitting data at %d with %s" %(i_split, split))

		if i_split > 0:
			x_train = self.x_data.iloc[0:i_split,0:].values
			y_train = self.y_data.iloc[0:i_split,0:].values

			x_test = self.x_data.iloc[i_split:,0:].values
			y_test = self.y_data.iloc[i_split:,0:].values

			return (x_train, y_train, x_test, y_test)
		elif i_split == 0:
			x_data = self.x_data.iloc[0:,0:].values
			y_data = self.y_data.iloc[0:,0:].values

			return (x_data, y_data)

	def __getitem__(self, index):
		
		x = self.x_data.iloc[index,0:].values.astype(np.float).reshape(1,self.x_data.shape[1])
		y  = self.y_data.iloc[index,0]

		return	x, y

	def __len__(self):
		return self.len

	def inverse_transform(self, x):
		return self.scaler.inverse_transform(x)