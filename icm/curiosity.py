import torch
import torch.nn as nn
from .models import *

class CuriosityModule(nn.Module):
	def __init__(self, input_shape, hidden_filters, num_layers, inv_hidden_dim, fwd_hidden_dim, action_dim, device, gamma=0.1, beta=0.2, reward_multiplier=5):
		super(CuriosityModule, self).__init__()
		in_filters, state_h, state_w = input_shape
		final_h = ((((((((state_h - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1)
		final_w = ((((((((state_w - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1)
		feature_dim = hidden_filters * final_h * final_w
		
		self.featurizer = Featurizer(in_filters, hidden_filters, num_layers, feature_dim).to(device)
		self.inv_model = InverseModel(feature_dim, action_dim, inv_hidden_dim).to(device)
		self.fwd_model = ForwardModel(feature_dim, action_dim, fwd_hidden_dim).to(device)
		self.beta = beta
		self.gamma = gamma
		self.reward_multiplier = reward_multiplier
		self.a_t_pred_loss = nn.CrossEntropyLoss()

	def forward(self, s_t, s_tp1, a_t):
		"""
		Returns loss value for this module as well as the intrinsic reward and predicted action logits.
		"""
		f_t = self.featurizer(s_t)
		f_tp1 = self.featurizer(s_tp1)

		a_t_logits = self.inv_model(f_t, f_tp1)
		f_tp1_pred = self.fwd_model(f_t, a_t)

		assert(a_t_logits.size(0) == a_t.size(0))

		L_I = torch.mean(self.a_t_pred_loss(a_t_logits, a_t))
		L_F = torch.mean(torch.div(torch.sum(torch.pow(f_tp1_pred - f_tp1, 2), dim=1), 2))
		with torch.no_grad():
			r_t = self.reward_multiplier * L_F
		return (1 - self.beta) * L_I + self.beta * L_F, r_t