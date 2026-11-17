import os, sys
TR = sys.argv[1]                       # path to the training/ folder
f2 = os.path.join(TR, "features2.py"); tv = os.path.join(TR, "train_v4.py")
s = open(f2).read()
if "def make_model" not in s:
    s += '''

def make_model(arch="gru", n_in=N_FEATURES2, hidden=64):
    """Sequence models with the same interface: (batch, time, n_in) -> (batch, time, 3) logits."""
    import torch, torch.nn as nn
    if arch == "gru":
        return make_gru2(hidden, n_in)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 3)
        def forward(self, x):
            return self.head(self.rnn(x)[0])

    class Block(nn.Module):
        def __init__(self, c, dil):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
            self.norm = nn.GroupNorm(1, c)
            self.drop = nn.Dropout(0.1)
        def forward(self, x):
            return x + self.drop(torch.relu(self.norm(self.conv(x))))

    class TCN(nn.Module):          # dilated non-causal convolutions, receptive field about 125 steps (25 s)
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, hidden, 1)
            self.blocks = nn.Sequential(*[Block(hidden, d) for d in (1, 2, 4, 8, 16)])
            self.head = nn.Conv1d(hidden, 3, 1)
        def forward(self, x):
            return self.head(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    class ConvGRU(nn.Module):      # local conv smoothing in front of a bidirectional GRU
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.ReLU())
            self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 3)
        def forward(self, x):
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return self.head(self.rnn(h)[0])

    class Transformer(nn.Module):  # conv embedding + 3 self-attention layers over the 60 s window
        def __init__(self, win=300):
            super().__init__()
            self.emb = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.ReLU())
            self.pos = nn.Parameter(torch.zeros(1, win, hidden))
            layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, 0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 3)
            self.head = nn.Linear(hidden, 3)
        def forward(self, x):
            h = self.emb(x.transpose(1, 2)).transpose(1, 2)
            return self.head(self.enc(h + self.pos[:, :h.shape[1]]))

    return {"lstm": LSTM, "tcn": TCN, "convgru": ConvGRU, "transformer": Transformer}[arch]()
'''
    open(f2, "w").write(s)
t = open(tv).read()
if "--arch" not in t:
    t = t.replace("from features2 import features2, make_gru2", "from features2 import features2, make_model")
    t = t.replace("    model = make_gru2(n_in=N_IN)\n", "    model = make_model(ARCH, n_in=N_IN)\n")
    t = t.replace("N_IN = 35 ", "ARCH = \"gru\"\nN_IN = 35 ", 1)
    t = t.replace('    ap.add_argument("--tag", default="v4")\n', '    ap.add_argument("--tag", default="v4")\n    ap.add_argument("--arch", default="gru", choices=["gru", "lstm", "tcn", "convgru", "transformer"])\n')
    t = t.replace("    global N_IN\n    N_IN = args.n_in\n", "    global N_IN, ARCH\n    N_IN = args.n_in\n    ARCH = args.arch\n")
    t = t.replace('print("RESULT tag={} n_in={}', 'print("RESULT arch={} tag={} n_in={}').replace("        args.tag, args.n_in, args.no_st,", "        args.arch, args.tag, args.n_in, args.no_st,")
    open(tv, "w").write(t)
print("patched", TR)
