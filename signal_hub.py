import sqlite3

class SignalHub:
    def __init__(self):
        self.conn = sqlite3.connect('comm_ledger.db')
        self.conn.execute('CREATE TABLE IF NOT EXISTS logs (time TEXT, shape TEXT)')

    def process(self, tensor, source):
        tensor = tensor.view(1, -1)  # Standardize: [1, features]
        self.conn.execute('INSERT INTO logs VALUES (?, ?)', (str(time.time()), str(tensor.shape)))
        self.conn.commit()
        return tensor

hub = SignalHub()
mlp_out = mlp(input_tensor)
hub.process(mlp_out, 'mlp')