# app.py — Urdu → Roman Urdu transliteration (WordPiece Seq2Seq BiLSTM-LSTM)

import streamlit as st
import torch
import torch.nn as nn
import pickle, re
from tokenizers import BertWordPieceTokenizer

# -------------------------------
# 1) Utility functions
# -------------------------------
@st.cache_resource
def load_tokenizers_and_vocabs():
    # Load trained WordPiece tokenizers
    src_tokenizer = BertWordPieceTokenizer("tokenizer_src/vocab.txt", lowercase=False)
    tgt_tokenizer = BertWordPieceTokenizer("tokenizer_tgt/vocab.txt", lowercase=True)

    # Load vocabularies (token -> id dicts)
    with open("src_tokenizer_vocab.pkl", "rb") as f:
        src2id = pickle.load(f)
    with open("tgt_tokenizer_vocab.pkl", "rb") as f:
        tgt2id = pickle.load(f)

    id2src = {v: k for k, v in src2id.items()}
    id2tgt = {v: k for k, v in tgt2id.items()}

    return src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt


def normalize_urdu(text):
    if not text: return ""
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ى', 'ی')
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s٬،۔؟]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# 2) Model Definitions
# -------------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

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
        h0 = torch.stack(h_list, dim=0)
        c0 = torch.stack(c_list, dim=0)
        return out, (h0, c0)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
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
        # not used in inference mode
        pass


# -------------------------------
# 3) Load Model + Artifacts
# -------------------------------
@st.cache_resource
def load_model():
    src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt = load_tokenizers_and_vocabs()

    PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
    src_pad_id = src2id.get(PAD, 0)
    tgt_pad_id = tgt2id.get(PAD, 0)
    src_bos_id = src2id.get(BOS, 1)
    src_eos_id = src2id.get(EOS, 2)
    tgt_bos_id = tgt2id.get(BOS, 1)
    tgt_eos_id = tgt2id.get(EOS, 2)

    EMB_DIM, HID_DIM = 256, 512
    ENC_LAYERS, DEC_LAYERS = 2, 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = BiLSTMEncoder(len(src2id), EMB_DIM, HID_DIM, ENC_LAYERS, 0.3, src_pad_id)
    dec = LSTMDecoder(len(tgt2id), EMB_DIM, HID_DIM, DEC_LAYERS, 0.3, tgt_pad_id)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    # Load model weights
    model.load_state_dict(torch.load("model_weights_wordpiece.pth", map_location=DEVICE))
    model.eval()

    return model, src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2src, id2tgt, DEVICE


# -------------------------------
# 4) Inference function
# -------------------------------
def greedy_translate(model, sentence, src_tokenizer, tgt_tokenizer, src2id, tgt2id, id2tgt, device, max_len=150):
    s = normalize_urdu(sentence)
    if not s.strip():
        return ""

    PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
    src_bos_id = src2id[BOS]
    src_eos_id = src2id[EOS]
    tgt_bos_id = tgt2id[BOS]
    tgt_eos_id = tgt2id[EOS]
    tgt_pad_id = tgt2id[PAD]

    enc = src_tokenizer.encode(s)
    s_ids = [src_bos_id] + enc.ids + [src_eos_id]
    if len(s_ids) > max_len:
        s_ids = s_ids[:max_len]
        s_ids[-1] = src_eos_id

    src_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_lens = torch.tensor([len(s_ids)], dtype=torch.long).to(device)

    _, (h_n, c_n) = model.encoder(src_tensor, src_lens)
    dec_nlayers = model.decoder.lstm.num_layers
    if h_n.size(0) < dec_nlayers:
        diff = dec_nlayers - h_n.size(0)
        h_n = torch.cat([h_n, h_n[-1:].repeat(diff, 1, 1)], dim=0)
        c_n = torch.cat([c_n, c_n[-1:].repeat(diff, 1, 1)], dim=0)

    input_tok = torch.tensor([tgt_bos_id], dtype=torch.long).to(device)
    hidden, cell = h_n, c_n
    out_ids = []
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_tok, hidden, cell)
        top1 = logits.argmax(1).item()
        if top1 == tgt_eos_id:
            break
        if top1 not in [tgt_pad_id, tgt_bos_id]:
            out_ids.append(top1)
        input_tok = torch.tensor([top1], dtype=torch.long).to(device)

    try:
        text = tgt_tokenizer.decode(out_ids, skip_special_tokens=True)
        return text.strip()
    except Exception:
        tokens = [id2tgt.get(i, "[UNK]") for i in out_ids]
        joined = " ".join(tokens).replace(" ##", "").replace("##", "")
        return re.sub(r"\s+", " ", joined).strip()


# -------------------------------
# 5) Streamlit UI
# -------------------------------
st.set_page_config(page_title="Urdu → Roman Urdu Transliteration", page_icon="🌀", layout="centered")

st.title("🌀 Urdu → Roman Urdu Transliteration")
st.markdown("Enter Urdu text below to get its Roman Urdu transliteration:")

urdu_input = st.text_area("Enter Urdu text:", height=150, placeholder="میں تم سے محبت کرتا ہوں")

if st.button("🔁 Transliterate"):
    with st.spinner("Transliterating..."):
        model, src_tok, tgt_tok, src2id, tgt2id, id2src, id2tgt, DEVICE = load_model()
        roman_text = greedy_translate(model, urdu_input, src_tok, tgt_tok, src2id, tgt2id, id2tgt, DEVICE)
        if roman_text.strip():
            st.success("**Roman Urdu Transliteration:**")
            st.markdown(f"<div style='padding:10px;border-radius:10px;background-color:#f0f0f0;font-size:18px;'>{roman_text}</div>", unsafe_allow_html=True)
        else:
            st.warning("Could not generate transliteration — please check your input.")
