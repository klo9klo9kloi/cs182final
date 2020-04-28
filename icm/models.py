import torch
import torch.nn as nn

class InverseModel(nn.Module):
	def __init__(self, in_filters, hidden_filters, num_layers, feature_dim, action_dim):
		super(InverseModel, self).__init__()
		layers = [nn.Conv2d(in_filters, hidden_filters, 3, stride=2, padding=1), nn.ELU(inplace=True)]
		for i in range(num_layers-1):
			layers.append(nn.Conv2d(hidden_filters, hidden_filters, 3, stride=2, padding=1))
			layers.append(nn.ELU(inplace=True))

		self.featurizer = nn.Sequential(*layers)

		self.predictor = nn.Sequential(
			nn.Linear(2*feature_dim, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, action_dim)
			)

	def forward(self, s_t, s_tp1):
		batch_size = s_t.size(0)
		f_t = self.featurizer(s_t).reshape(batch_size, -1)
		f_tp1 = self.featurizer(s_tp1).reshape(batch_size, -1)

		action_logits = self.predictor(torch.cat([f_t, f_tp1], dim=1))
		return action_logits, f_t

class ForwardModel(nn.Module):
	def __init__(self, feature_dim, action_dim, hidden_dim):
		super(ForwardModel, self).__init__()
		self.predictor = nn.Sequential(
			nn.Linear(feature_dim+action_dim, hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_dim, feature_dim)
		)
	def forward(self, f_t, a_t):
		return self.predictor(torch.cat([f_t, a_t], dim=1))



