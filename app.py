# app.py — Streamlit Urdu→Roman Urdu Transliteration using WordPiece-BiLSTM model

import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from tokenizers import BertWordPieceTokenizer

# -------------------------------
# Config
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 150

# -------------------------------
# Load Tokenizers and Vocab
# -------------------------------
@st.cache_resource
def load_tokenizers_and_vocabs():
    src_tokenizer = BertWordPieceTokenizer("tokenizer_src/vocab.txt", lowercase=False)
    tgt_tokenizer = BertWordPieceTokenizer("tokenizer_tgt/vocab.txt", lowercase=True)

    with open("src_tokenizer_vocab.pkl", "rb") as f:
        src2id = pickle.load(f)
    with open("tgt_tokenizer_vocab.pkl", "rb") as f:
        tgt2id = pickle.load(f)

    id2src = {v: k for k, v in src2id.items()}
    id2tgt = {v: k for k, v in tgt2id.items()}

    return src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt

src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt = load_tokenizers_and_vocabs()

# Special tokens
PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
src_pad_id = src2id[PAD]
src_bos_id = src2id[BOS]
src_eos_id = src2id[EOS]
tgt_pad_id = tgt2id[PAD]
tgt_bos_id = tgt2id[BOS]
tgt_eos_id = tgt2id[EOS]

# -------------------------------
# Model Definition
# -------------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h_list, c_list = [], []
        for i in range(self.n_layers):
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
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        pass  # not used in inference


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    EMB_DIM = 256
    HID_DIM = 512
    enc = BiLSTMEncoder(len(src2id), EMB_DIM, HID_DIM, n_layers=2, dropout=0.3, pad_idx=src_pad_id)
    dec = LSTMDecoder(len(tgt2id), EMB_DIM, HID_DIM, n_layers=4, dropout=0.3, pad_idx=tgt_pad_id)
    model = Seq2Seq(enc, dec).to(DEVICE)

    state_dict = torch.load("model_weights_wordpiece.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Helper Functions
# -------------------------------
def normalize_urdu(text):
    if not text: return ""
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ی')
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s٬،۔؟]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ids_to_text(ids):
    tokens = [id2tgt.get(i, "[UNK]") for i in ids if i not in [tgt_bos_id, tgt_eos_id, tgt_pad_id]]
    try:
        text = tgt_tokenizer.decode(ids, skip_special_tokens=True)
        return text.strip()
    except:
        return " ".join(tokens).replace(" ##", "").replace("##", "").strip()

def translate(sentence):
    model.eval()
    with torch.no_grad():
        s = normalize_urdu(sentence)
        enc = src_tokenizer.encode(s)
        s_ids = [src_bos_id] + enc.ids + [src_eos_id]
        if len(s_ids) > MAX_LEN:
            s_ids = s_ids[:MAX_LEN]
            s_ids[-1] = src_eos_id
        src_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        src_lens = torch.tensor([len(s_ids)], dtype=torch.long).to(DEVICE)
        _, (h_n, c_n) = model.encoder(src_tensor, src_lens)

        dec_nlayers = model.decoder.lstm.num_layers
        enc_nlayers = h_n.size(0)
        if enc_nlayers < dec_nlayers:
            h_n = torch.cat([h_n, h_n[-1:].repeat(dec_nlayers - enc_nlayers, 1, 1)], dim=0)
            c_n = torch.cat([c_n, c_n[-1:].repeat(dec_nlayers - enc_nlayers, 1, 1)], dim=0)

        input_tok = torch.tensor([tgt_bos_id], dtype=torch.long).to(DEVICE)
        hidden, cell = h_n, c_n
        out_ids = []
        for _ in range(MAX_LEN):
            logits, hidden, cell = model.decoder(input_tok, hidden, cell)
            top1 = logits.argmax(1).item()
            if top1 == tgt_eos_id:
                break
            if top1 not in [tgt_pad_id, tgt_bos_id]:
                out_ids.append(top1)
            input_tok = torch.tensor([top1], dtype=torch.long).to(DEVICE)
        return ids_to_text(out_ids)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌀 Urdu → Roman Urdu Transliteration", layout="centered")

st.title("🌀 Urdu → Roman Urdu Transliteration")
st.markdown("Enter Urdu text below to get its Roman Urdu transliteration:")

urdu_text = st.text_area("Enter Urdu text:", height=150, placeholder="مثلاً: میں تم سے محبت کرتا ہوں")

if st.button("🔁 Transliterate"):
    if urdu_text.strip():
        with st.spinner("Generating Roman transliteration..."):
            output = translate(urdu_text)
        st.success("✅ Transliteration complete!")
        st.text_area("Roman Urdu Output:", value=output, height=150)
    else:
        st.warning("Please enter some Urdu text above.")

st.markdown("---")
st.markdown("👨‍💻 *Powered by BiLSTM + LSTM (WordPiece)*")
