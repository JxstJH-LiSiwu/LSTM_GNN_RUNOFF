from src.lstm_only import CombinedLSTMWithStatic2Hop as LSTMOnly
from src.lstm_gat import CombinedLSTMWithStatic2Hop as LSTMGAT
from src.lstm_gcn import CombinedLSTMWithStatic2Hop as LSTMGCN
from src.lstm_cheb import CombinedLSTMWithStatic2Hop as LSTMCheb
from src.lstm_sage import CombinedLSTMWithStatic2Hop as LSTMSage

MODEL_REGISTRY = {
    "LSTM": {"class": LSTMOnly, "ckpt": "lstm_only_model_epoch.pth"},
    "LSTM-GAT": {"class": LSTMGAT, "ckpt": "lstm_gat_model_epoch.pth"},
    "LSTM-GCN": {"class": LSTMGCN, "ckpt": "lstm_gcn_model_epoch.pth"},
    "LSTM-Cheb": {"class": LSTMCheb, "ckpt": "lstm_cheb_model_epoch.pth"},
    "LSTM-GraphSAGE": {"class": LSTMSage, "ckpt": "lstm_sage_model_epoch.pth"},
}
