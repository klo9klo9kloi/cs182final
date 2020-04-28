import torch
import torch.nn as nn
from .models import *

class CuriosityModule():
	def __init__(self, input_shape, inv_hidden_filters, fwd_hidden_dim, inv_num_layers, action_dim, gamma=0.1, beta=0.2, reward_multiplier=5):
		in_filters, state_h, state_w = input_shape
		final_h = ((((((((state_h - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1)
		final_w = ((((((((state_w - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1) - 1)//2 + 1)
		feature_dim = inv_hidden_filters * final_h * final_w
		
		self.inv_model = InverseModel(in_filters, inv_hidden_filters, inv_num_layers, feature_dim, action_dim)
		self.fwd_model = ForwardModel(feature_dim, action_dim, fwd_hidden_dim)
		self.beta = beta
		self.gamma = gamma
		self.reward_multiplier = reward_multiplier
		self.a_t_pred_loss = nn.CrossEntropyLoss()

	def forward(self, s_t, s_tp1, a_t):
		"""
		Returns loss value for this module as well as the intrinsic reward and predicted action logits.
		"""
		a_t_logits, f_t = self.inv_model(s_t, s_tp1)
		s_tp1_pred = self.fwd_model(f_t, a_t)

		assert(a_t_logits.size(0) == a_t.size(0))

		L_I = torch.mean(self.a_t_pred_loss(a_t_logits, a_t))
		L_F = torch.div(torch.pow(s_tp1_pred - s_tp1, 2), 2)
		r_t = self.reward_multiplier * L_F.data.cpu().numpy()
		return (1 - self.beta) * L_I + self.beta * L_F, r_t