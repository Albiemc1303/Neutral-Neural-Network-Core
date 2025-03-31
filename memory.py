class HoloMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2000, 2000, 1)  # 2K cells
        self.caps = nn.Conv2d(1, 2000, 3)   # Dummy CapsNets
        self.reservoir = nn.Linear(5000, 5000)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(1, 1, -1))
        caps_out = self.caps(x.view(1, 1, 45, 45))  # Fake 45x45 input
        res_out = self.reservoir(lstm_out.view(1, -1))
        return torch.cat([lstm_out, caps_out.flatten(1), res_out], dim=1)

memory = HoloMemory()
memory.lstm.to('cpu')
memory.caps.to('cuda')
memory.reservoir.to('cpu')