import torch
import torch.nn as nn
import torch.nn.functional as F
from .glimpse_modules import GlimpseNetwork

# Sanity test
# -----------
# import torch
# from models.models import model_with_glimpse
# q_func = model_with_glimpse(12, 15)
# q_func = q_func.to("cuda:0")
# img = torch.randn(5, 12, 64, 64).to("cuda:0")
# a_t = q_func(img)
# print(a_t.shape)

class base_atari_model(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(base_atari_model, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.action_value = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim)
        )
    def forward(self, x):
        x = self.convnet(x)
        x = x.reshape(x.shape[0], -1)
        x = self.action_value(x)
        return x

class model_with_glimpse(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(model_with_glimpse, self).__init__()
        if input_channels > 3:
            self.seq_len = input_channels // 3

        self.glimpse_network = GlimpseNetwork(6, 4, [512, 256, 512], 3, 2)

        self.location = nn.LSTM(512, 256) 
        self.action = nn.LSTM(512, 512)

        self.location_value = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2),
            nn.Tanh())

        self.action_value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim)
        )

        self.context = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.context_to_hidden = nn.Sequential(
            nn.Linear(64*4*4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        device = x.device

        a_hidden, l_hidden = self.init_hidden(x, device)

        sequence = torch.chunk(x, self.seq_len, dim=1)

        l_0, l_hidden = self.location(torch.zeros(1, x.size(0), 512).to(device), l_hidden)
        l_t = self.location_value(l_0[0])

        for x_t in sequence:
            g_t = self.glimpse_network(x_t, l_t)
            a, a_hidden = self.action(g_t.unsqueeze(0), a_hidden)
            l, l_hidden = self.location(a, l_hidden)
            l_t = self.location_value(l[0])

        a_t = self.action_value(a[0])
        return a_t

    def init_hidden(self, x, device):
        batch_size = x.size(0)
        action_hidden = (torch.zeros(1, batch_size, 512).to(device),
                  torch.zeros(1, batch_size, 512).to(device))

        location_hidden = self.context(x)
        location_hidden = location_hidden.reshape(batch_size, -1)
        location_hidden = self.context_to_hidden(location_hidden).unsqueeze(0).to(device)
        return action_hidden, (location_hidden, torch.zeros(1, batch_size, 256).to(device))

