import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from tokenizers import BertWordPieceTokenizer

# ---------------------------
# 1) Load tokenizers & vocabs
# ---------------------------
@st.cache_resource
def load_tokenizers_and_vocabs():
    # ensure vocab has standard tokens
    def ensure_special_tokens(vocab_path):
        required = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        missing = [t for t in required if t not in lines]
        if missing:
            with open(vocab_path, "w", encoding="utf-8") as f:
                for t in required:
                    f.write(t + "\n")
                for l in lines:
                    f.write(l + "\n")

    ensure_special_tokens("tokenizer_src/vocab.txt")
    ensure_special_tokens("tokenizer_tgt/vocab.txt")

    src_tokenizer = BertWordPieceTokenizer("tokenizer_src/vocab.txt", lowercase=False)
    tgt_tokenizer = BertWordPieceTokenizer("tokenizer_tgt/vocab.txt", lowercase=True)

    with open("src_tokenizer_vocab.pkl", "rb") as f:
        src2id = pickle.load(f)
    with open("tgt_tokenizer_vocab.pkl", "rb") as f:
        tgt2id = pickle.load(f)

    id2src = {v: k for k, v in src2id.items()}
    id2tgt = {v: k for k, v in tgt2id.items()}
    return src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt

# ---------------------------
# 2) Model definition
# ---------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h_list, c_list = [], []
        for i in range(self.lstm.num_layers):
            h_f, h_b = h_n[2*i], h_n[2*i+1]
            c_f, c_b = c_n[2*i], c_n[2*i+1]
            h = torch.tanh(self.fc_hidden(torch.cat((h_f, h_b), dim=1)))
            c = c_f + c_b
            h_list.append(h)
            c_list.append(c)
        return out, (torch.stack(h_list), torch.stack(c_list))

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell):
        emb = self.dropout(self.embedding(input_tok)).unsqueeze(1)
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        pass  # not used during inference

# ---------------------------
# 3) Helper functions
# ---------------------------
def normalize_urdu(text):
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ی')
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s٬،۔؟]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------------
# 4) Load model + resources
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM, HID_DIM, ENC_LAYERS, DEC_LAYERS = 256, 512, 2, 4

src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt = load_tokenizers_and_vocabs()

PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
src_pad_id = src2id[PAD]
tgt_pad_id, tgt_bos_id, tgt_eos_id = tgt2id[PAD], tgt2id[BOS], tgt2id[EOS]

enc = BiLSTMEncoder(len(src2id), EMB_DIM, HID_DIM, n_layers=ENC_LAYERS, pad_idx=src_pad_id)
dec = LSTMDecoder(len(tgt2id), EMB_DIM, HID_DIM, n_layers=DEC_LAYERS, pad_idx=tgt_pad_id)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.encoder = enc
model.decoder = dec
model.load_state_dict(torch.load("model_weights_wordpiece.pth", map_location=DEVICE))
model.eval()

# ---------------------------
# 5) Greedy decoding
# ---------------------------
MAX_LEN = 150
def ids_to_text_from_tgt(ids):
    try:
        return tgt_tokenizer.decode(ids, skip_special_tokens=True).strip()
    except:
        tokens = [id2tgt.get(i, "[UNK]") for i in ids if i not in [tgt_bos_id, tgt_eos_id, tgt_pad_id]]
        return re.sub(r'\s+', ' ', " ".join(tokens).replace("##", "")).strip()

def greedy_translate(sentence):
    s = normalize_urdu(sentence)
    enc_res = src_tokenizer.encode(s)
    s_ids = [src2id[BOS]] + enc_res.ids + [src2id[EOS]]
    src_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([len(s_ids)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        _, (h_n, c_n) = model.encoder(src_tensor, src_lens)
        dec_nlayers = model.decoder.lstm.num_layers
        if h_n.size(0) < dec_nlayers:
            pad_h = h_n[-1].unsqueeze(0).repeat(dec_nlayers - h_n.size(0), 1, 1)
            pad_c = c_n[-1].unsqueeze(0).repeat(dec_nlayers - c_n.size(0), 1, 1)
            h_n = torch.cat([h_n, pad_h], 0)
            c_n = torch.cat([c_n, pad_c], 0)
        input_tok = torch.tensor([tgt_bos_id], dtype=torch.long).to(DEVICE)
        hidden, cell = h_n, c_n
        out_ids = []
        for _ in range(MAX_LEN):
            logits, hidden, cell = model.decoder(input_tok, hidden, cell)
            top1 = logits.argmax(1).item()
            if top1 == tgt_eos_id: break
            if top1 not in [tgt_pad_id, tgt_bos_id]:
                out_ids.append(top1)
            input_tok = torch.tensor([top1], dtype=torch.long).to(DEVICE)
    return ids_to_text_from_tgt(out_ids)

# ---------------------------
# 6) Streamlit UI
# ---------------------------
st.title("🌀 Urdu → Roman Urdu Transliteration")
st.markdown("Enter Urdu text to get its Roman Urdu transliteration using a BiLSTM + WordPiece model.")

urdu_text = st.text_area("Enter Urdu text below:", "", height=150)
if st.button("Transliterate"):
    if urdu_text.strip():
        with st.spinner("Transliterating..."):
            roman_text = greedy_translate(urdu_text)
        st.success("✅ Transliteration Complete:")
        st.text_area("Roman Urdu Output:", roman_text, height=150)
    else:
        st.warning("Please enter some Urdu text.")
