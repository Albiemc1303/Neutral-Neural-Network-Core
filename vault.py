class AII(nn.Module):
    def __init__(self):
        super().__init__()
        self.chat = nn.Linear(5000, 5000)  # Simple "Grok"
        self.cpn = nn.Sequential(nn.Linear(2000, 2000), nn.Linear(2000, 3000))

    def forward(self, x):
        chat_out = self.chat(x)
        cpn_out = self.cpn(chat_out[:2000])
        return torch.cat([chat_out, cpn_out], dim=1)

me = AII().to('cpu')