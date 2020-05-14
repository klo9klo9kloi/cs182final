import torch
import torch.nn as nn

class InverseModel(nn.Module):
	def __init__(self, feature_dim, action_dim, hidden_dim):
		super(InverseModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(2*feature_dim, hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_dim, action_dim)
			)

	def forward(self, f_t, f_tp1):
		return self.model(torch.cat([f_t, f_tp1], dim=1))

class Featurizer(nn.Module):
	def __init__(self, in_filters, hidden_filters, num_layers, feature_dim):
		super(Featurizer, self).__init__()
		layers = [nn.Conv2d(in_filters, hidden_filters, 3, stride=2, padding=1), nn.ELU(inplace=True)]
		for i in range(num_layers-1):
			layers.append(nn.Conv2d(hidden_filters, hidden_filters, 3, stride=2, padding=1))
			layers.append(nn.ELU(inplace=True))

		self.model = nn.Sequential(*layers)

	def forward(self, s):
		batch_size = s.size(0)
		return self.model(s).reshape(batch_size, -1)

class ForwardModel(nn.Module):
	def __init__(self, feature_dim, action_dim, hidden_dim):
		super(ForwardModel, self).__init__()
		self.predictor = nn.Sequential(
			nn.Linear(feature_dim+action_dim, hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_dim, feature_dim)
		)
		self.one_hots = torch.eye(action_dim)
	def forward(self, f_t, a_t):
		return self.predictor(torch.cat([f_t, self.one_hots[a_t.cpu().numpy()].to(f_t.device)], dim=1))



