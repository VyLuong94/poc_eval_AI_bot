
import warnings
warnings.filterwarnings("ignore")
import nest_asyncio
import asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
import json
import hashlib
from collections import Counter
import openai
import httpx
import torch
import io
import threading
from io import BytesIO
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,RobertaForSequenceClassification, pipeline, AutoModelForTokenClassification,AutoModelForQuestionAnswering
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.language_models.llms import LLM
from sentence_transformers import SentenceTransformer,util
import time
import tracemalloc
import gc
import objgraph
from memory_profiler import profile
from pydub import AudioSegment
from pydub.utils import make_chunks
from transformers import pipeline
import math
import asyncio
import sys
import concurrent.futures
import nest_asyncio
import mimetypes
from collections import defaultdict, Counter
from unidecode import unidecode
nest_asyncio.apply()



if sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


def get_file_hash(file):
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

def transcribe_audio(uploaded_file):

    cache_dir = "transcripts"
    os.makedirs(cache_dir, exist_ok=True)

    file_hash = get_file_hash(uploaded_file)
    cache_path = os.path.join(cache_dir, f"{file_hash}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return cached_data["text"]

    uploaded_file.seek(0)
    filename = getattr(uploaded_file, 'name', 'audio.wav')
    content_type = mimetypes.guess_type(filename)[0] or 'audio/wav'

    transcription = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=(filename, uploaded_file, content_type),
        language="vi",
        temperature=0
    )


    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"text": transcription.text}, f, ensure_ascii=False, indent=2)

    return transcription.text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\(.*\)", " ", text)
    text = re.sub(r"<.*>", " ", text)
    text = re.sub(r"\[.*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = text.strip()

    return text


def detect_intent(text):
    text_lower = text.lower()

    intent_patterns = {
        "Chưa thanh toán": [
            r"không có tiền", r"chưa có tiền", r"chưa trả được", r"chị không hứa",
            r"mượn hết rồi", r"không biết khi nào trả", r"đang kẹt tiền",
            r"tháng sau", r"cuối tháng", r"đợi lương", r"trễ lương", r"lương chưa về",
            r"bị giảm công", r"giảm lương", r"người ta chưa trả",
            r"tôi đang khó khăn", r"chưa thu xếp được", r"để tôi xem đã"
        ],
        "Trả một phần": [
            r"trả trước \d+", r"đóng trước \d+", r"chuyển trước \d+", r"gửi trước một phần",
            r"có 1 triệu", r"tạm ứng", r"gửi trước 1 phần"
        ],
        "Thanh toán": [
            r"được rồi", r"đồng ý", r"ok", r"sẽ cố gắng", r"thanh toán", r"sẽ gửi", r"đóng đủ",
            r"cảm ơn bạn đã thông báo", r"tôi sẽ trao đổi thêm với gia đình", r"vâng, tôi hiểu",
            r"tôi đang cố gắng sắp xếp", r"nhờ công ty hỗ trợ", r"cho tôi thêm thời gian"
        ],

        "Không thanh toán": [
            r"không nghe nữa", r"đừng gọi nữa", r"không rảnh", r"cúp máy đây",
            r"phiền quá", r"muốn làm gì làm", r"tôi không cần biết",
            r"không đóng", r"không thanh toán",
            r"tôi bận", r"tôi không biết", r"đừng gọi làm phiền tôi"
        ],
        "Yêu cầu giãn nợ": [
            r"giãn nợ", r"gia hạn", r"xin khất", r"chờ thêm vài ngày", r"xin thêm thời gian"
        ]
    }

    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent

    return "Không rõ"


def extract_sop_items_from_excel(file_path, sheet_name=0):
    if isinstance(file_path, bytes):
        df = pd.read_excel(BytesIO(file_path), sheet_name=sheet_name, header=1)
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    if 'Tên tiêu chí đánh giá ' in df.columns and 'Tên tiêu chí đánh giá' not in df.columns:
        df.rename(columns={'Tên tiêu chí đánh giá ': 'Tên tiêu chí đánh giá'}, inplace=True)

    required_columns = ['Mã tiêu chí', 'Tên tiêu chí đánh giá', 'Hướng dẫn thực hiện', 'Điểm', 'Hướng dẫn đánh giá']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Thiếu cột: {[col for col in required_columns if col not in df.columns]}")

    df.dropna(subset=required_columns, how='all', inplace=True)
    df = df[~df[required_columns].apply(lambda x: x.astype(str).str.strip().eq('').all(), axis=1)]

    df[['Tên tiêu chí đánh giá', 'Hướng dẫn thực hiện']] = df[['Tên tiêu chí đánh giá', 'Hướng dẫn thực hiện']].ffill()
    df['Mã tiêu chí'] = df['Mã tiêu chí'].ffill()
    df['Hướng dẫn đánh giá'] = df['Hướng dẫn đánh giá'].fillna("")
    df['Điểm'] = df['Điểm'].fillna(0)

    def ensure_dot_end(code):
        code = code.strip()
        if not code.endswith('.'):
            code += '.'
        return code

    child_codes = df['Mã tiêu chí'].astype(str)[df['Mã tiêu chí'].astype(str).str.count(r'\.') >= 2].apply(ensure_dot_end)
    parent_codes = child_codes.apply(lambda x: '.'.join(x.strip('.').split('.')[:2])).unique()


    df = df[~((df['Mã tiêu chí'].astype(str).str.rstrip('.')\
                .isin(parent_codes)) &
              (df['Mã tiêu chí'].astype(str).str.count(r'\.') == 1))]

    df = df[df['Điểm'] != 0]

    sop_items = []
    current_section = None

    def format_instruction(text):
        lines = str(text).split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                formatted_lines.append('\n' + line)
            else:
                formatted_lines.append(line)
        return ' '.join(formatted_lines).strip()


    for _, row in df.iterrows():
        code = str(row['Mã tiêu chí']).strip().rstrip('.')

        if not re.match(r'^\d+(\.\d+)*$', code):
            current_section = {
                "section": clean_text_sop(row['Tên tiêu chí đánh giá']),
                "items": []
            }
            sop_items.append(current_section)
            continue

        title = clean_text_sop(row['Tên tiêu chí đánh giá'])
        raw_instruction = format_instruction(row['Hướng dẫn thực hiện'])
        instruction_lines = [
            clean_text_sop(line.strip())
            for line in raw_instruction.split('\n')
            if clean_text_sop(line.strip())
        ]

        filtered_instructions = [line for line in instruction_lines if line != title]

        if not title:
            continue

        description = "\n".join(filtered_instructions) if filtered_instructions else title
        score = int(float(row['Điểm'])) if pd.notnull(row['Điểm']) else 0

        current_section['items'].append({
            "code": code,
            "title": title,
            "description": description,
            "score": score
        })


    return sop_items


def clean_text_sop(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'"[^"]*"', '', text)
    text = re.sub(r'^\s*\d+([\.\-\d]*)[\)\.\-\s]*', '', text)
    text = re.sub(r'[;:()\[\]{}"]', '', text)
    text = ' '.join(text.split())
    text = text.replace('\xa0', ' ').strip()
    if not text or re.fullmatch(r'\d+|[^\w\s\-]+', text):
        return ""

    return text.strip()


def split_sop_into_subsentences(text):
    if not isinstance(text, str) or not text.strip():
        return []

    lines = text.strip().split('\n')
    results = []

    for line in lines:
        cleaned = clean_text_sop(line.strip())
        if cleaned:
            results.append(cleaned)

    return results



IGNORE_KEYWORDS = [
    "alo", "chào", "em gọi", "cho em hỏi", "không ạ", "bên em", "đơn vị", "công ty", "em là", "gọi cho chị", "từ bên", "liên kết",
    "chậm nhất", "thanh toán", "hồ sơ", "ngân hàng", "báo cáo", "giùm em", "hả", "xin phép gọi lại sau", "nói với", "nhờ chị",
    "không biết là", "báo cho chị", "chuyển luôn cho em", "Đúng rồi", "liên lạc lại sau", "đúng không", "chưa chị", "chị tính",
    "mình đóng", "em báo", "nhắc", "xử lý", "thu", "vậy ạ", "dạ không đổi số", "giúp nha", "giúp", "xin phép", "đóng giúp", "mình bảo",
    "em cảm ơn", "trễ", "em bên"
]


def is_greeting_or_intro(sentence):
    sentence_lower = sentence.lower()
    return any(kw in sentence_lower for kw in IGNORE_KEYWORDS)


def split_transcript_into_sentences(text):

    if not text or not isinstance(text, str):
        return []

    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    subsentences = []
    buffer = ""

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        cleaned = re.sub(r"[;\-\•]", ".", sentence)
        small_parts = [s.strip() for s in cleaned.split('.') if s.strip()]

        for part in small_parts:
            if len(part.split()) <= 3:
                buffer += " " + part
            else:
                if buffer:
                    subsentences.append((buffer + " " + part).strip())
                    buffer = ""
                else:
                    subsentences.append(part)
    if buffer:
        subsentences.append(buffer.strip())

    return subsentences

def detect_sheet_from_text(agent_text):
    if not isinstance(agent_text, str):
        return 'Tiêu chí giám sát cuộc gọi KH'

    agent_text = agent_text.lower()

    patterns = [
    r"cho hỏi.*(vợ|chồng|con|ba|mẹ|người nhà|người thân).*có phải.*đang nghe máy",
    r"(chị|anh|em) có phải là.*(vợ|chồng|con|người thân) của",
    r"(vợ|chồng|con) của (anh|chị)",
    r"(chị|anh|em) là (vợ|chồng|người thân|người nhà) của",
    r"(vợ|chồng|con).*đang nghe máy",
    r"xin phép.*liên hệ.*(người nhà|người thân)",
]

    if any(re.search(p, agent_text) for p in patterns):
        return 'Tiêu chí giám sát cuộc gọi NT'

    return 'Tiêu chí giám sát cuộc gọi KH'


@st.cache_resource
def load_ner_pipeline():
    """Load NER pipeline with Hugging Face token."""
    model_name = "NlpHUST/ner-vietnamese-electra-base"
    hf_api_token = st.secrets["huggingface"]["token"]

    try:

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_token)
        model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=hf_api_token)


        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            grouped_entities=True,
            device=0 if torch.cuda.is_available() else -1
        )

        return ner_pipeline

    except Exception as e:
        st.error(f"An error occurred while loading the NER pipeline: {e}")
        return None


ner_pipeline = load_ner_pipeline()

def merge_short_sentences(sentences, short_length=3):
    merged = []
    buffer = ""

    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) <= short_length:
            buffer += " " + sentence
        else:
            if buffer:
                merged.append((buffer + " " + sentence).strip())
                buffer = ""
            else:
                merged.append(sentence.strip())
    if buffer:
        merged.append(buffer.strip())
    return merged


def analyze_call_transcript(text, min_sentence_length=5):
    raw_sentences = re.split(r'(?<=[.!?]) +', text)
    merged_sentences = merge_short_sentences(raw_sentences)

    filtered_sentences = [
        s.strip() for s in merged_sentences
        if len(s.split()) >= min_sentence_length and not is_greeting_or_intro(s)
    ]

    entities_dict = defaultdict(set)
    cooperative_sentences = 0
    tone_counter = Counter()
    tone_chunks_result = []

    for sentence in filtered_sentences:
        try:
            tones = classify_tone(sentence)
        except Exception as e:
            tones = [{"text": sentence, "tone": "Error"}]

        if isinstance(tones, list):
            for tone in tones:
                tone_value = tone.get('tone', '') if isinstance(tone, dict) else tone
                tone_chunks_result.append({"text": sentence, "tone": tone_value})
                tone_counter[tone_value] += 1
                if tone_value == "Hợp tác":
                    cooperative_sentences += 1

        try:
            ner_results = ner_pipeline(sentence)
            for entity in ner_results:
                entities_dict[entity['entity_group']].add(entity['word'])
        except:
            continue

    try:
        intent_result = detect_intent(text)
    except:
        intent_result = "Không xác định"

    total_sentences = len(filtered_sentences)
    collaboration_rate = (cooperative_sentences / total_sentences) * 100 if total_sentences else 0

    interaction_summary = "=== Đánh giá tương tác ===\n"
    if collaboration_rate < 50:
        interaction_summary += f"Tỷ lệ hợp tác thấp ({collaboration_rate:.2f}%). Câu không hợp tác:\n"
        for i, item in enumerate(tone_chunks_result):
            if item["tone"] == "Không hợp tác":
                interaction_summary += f"{i+1}. {item['text']}\n"
    else:
        interaction_summary += f"Tỷ lệ hợp tác cao ({collaboration_rate:.2f}%). Câu hợp tác:\n"
        for i, item in enumerate(tone_chunks_result):
            if item["tone"] == "Hợp tác":
                interaction_summary += f"{i+1}. {item['text']}\n"

    return {
        "named_entities": {k: list(v) for k, v in entities_dict.items()},
        "intent": intent_result,
        "collaboration_rate": round(collaboration_rate, 2),
        "interaction_summary": interaction_summary,
        "tone_chunks": {
            "tone_summary": tone_counter,
            "important_chunks": [
                chunk for chunk in tone_chunks_result if chunk["tone"] in ["Hợp tác", "Không hợp tác"]
            ]
        }
    }


@st.cache_resource
def load_model():
    """Load tone classification model and tokenizer."""
    model_name = "vyluong/AI-poc-v4-tone-classification-model"
    hf_api_token = st.secrets["huggingface"]["token"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_token)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_auth_token=hf_api_token
        )
        model.to(device)
        model.eval()

        return tokenizer, model, device

    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None, None


@st.cache_resource
def load_excel_rag_data(uploaded_excel_file):
    try:
        class QA_LLM:
            def __init__(self, model_qa, tokenizer_qa, sop_text):
                self.model_qa = model_qa
                self.tokenizer_qa = tokenizer_qa
                self.sop_text = sop_text

            def _call(self, prompt: str, **kwargs) -> str:
                context = kwargs.get("context", self.sop_text)
                inputs = self.tokenizer_qa(prompt, context, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.model_qa.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model_qa(**inputs)

                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)

                start_conf = torch.softmax(outputs.start_logits, dim=-1)[start_index].item()
                end_conf = torch.softmax(outputs.end_logits, dim=-1)[end_index].item()

                if end_index < start_index or (end_index - start_index) > 50 or min(start_conf, end_conf) < 0.3:
                    return "Không tìm thấy thông tin phù hợp trong SOP."

                answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
                answer = self.tokenizer_qa.decode(answer_tokens, skip_special_tokens=True).strip()

                if len(answer.split()) <= 1:
                    return "Không tìm thấy thông tin phù hợp trong SOP."

                return answer


            @property
            def _llm_type(self) -> str:
                return "custom-bert-qa"

        if isinstance(uploaded_excel_file, str):
            with open(uploaded_excel_file, 'rb') as f:
                file_data = f.read()
            excel_file = BytesIO(file_data)
        elif hasattr(uploaded_excel_file, 'read'):
            excel_file = uploaded_excel_file
        else:
            print("No valid file uploaded.")
            return None, None, None, None

        xls = pd.ExcelFile(excel_file)
        available_sheets = xls.sheet_names
        required_sheets = ['Tiêu chí giám sát cuộc gọi KH', 'Tiêu chí giám sát cuộc gọi NT']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
        if missing_sheets:
            print(f"Missing sheets: {missing_sheets}")
            return None, None, None, None

        df_customer_call = pd.read_excel(excel_file, sheet_name='Tiêu chí giám sát cuộc gọi KH')
        df_relative_call = pd.read_excel(excel_file, sheet_name='Tiêu chí giám sát cuộc gọi NT')

        if df_customer_call.empty or df_relative_call.empty:
            print("One or both sheets are empty.")
            return None, None, None, None

        customer_docs = [
            Document(page_content=row_text, metadata={"sheet_name": "Tiêu chí giám sát cuộc gọi KH"})
            for row_text in df_customer_call.astype(str).agg(' '.join, axis=1).tolist()
        ]
        relative_docs = [
            Document(page_content=row_text, metadata={"sheet_name": "Tiêu chí giám sát cuộc gọi NT"})
            for row_text in df_relative_call.astype(str).agg(' '.join, axis=1).tolist()
        ]

        all_docs = customer_docs + relative_docs
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(split_docs, embedding_model)
        retriever = db.as_retriever()

        combined_text = "\n".join([doc.page_content for doc in all_docs])
        sop_data = {
            'Tiêu chí giám sát cuộc gọi KH': df_customer_call.astype(str).agg(' '.join, axis=1).tolist(),
            'Tiêu chí giám sát cuộc gọi NT': df_relative_call.astype(str).agg(' '.join, axis=1).tolist()
        }

        model_name_qa = "nguyenvulebinh/vi-mrc-large"
        tokenizer_qa = AutoTokenizer.from_pretrained(model_name_qa)
        model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name_qa)

        model_qa.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        qa_llm = QA_LLM(model_qa, tokenizer_qa, combined_text)


        return qa_llm, retriever, sop_data, combined_text

    except Exception as e:
        print(f"Error loading Excel data: {e}")
        return None, None, None, None


tokenizer, model, device = load_model()


def safe_tokenize(text, tokenizer, max_length=256):
    """
    Tokenize text with error handling.
    - Will skip text if error occurs during tokenization (e.g., long text, unexpected characters).
    """
    try:
        text = clean_text(text)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        return inputs

    except Exception as e:
        st.error(f"An error occurred during tokenization: {e}")
        return None


def classify_tone(text, chunk_size=None):
    """
    Classify tone of text.
    - If chunk_size is None: split into individual sentences and classify each.
    - If chunk_size is an integer: group sentences into chunks and classify each chunk with majority voting.
    """
    text = clean_text(text)

    def predict(inputs):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_id = inputs['input_ids'].max().item()
        if max_id >= model.config.vocab_size:
            return None

        with torch.no_grad():
            outputs = model(**inputs)
        label = outputs.logits.argmax(dim=1).cpu().item()
        return label

    if chunk_size is None:
        sentences = split_transcript_into_sentences(text)
        results = []

        for sentence in sentences:
            sentence_clean = clean_text(sentence)
            if not sentence_clean.strip():
                continue

            inputs = safe_tokenize(sentence_clean, tokenizer)
            if inputs is None:
                results.append({"text": sentence, "tone": "Error in tokenization"})
                continue

            label = predict(inputs)
            if label is None:
                results.append({"text": sentence, "tone": "Error: input_ids out of range"})
                continue

            tone = "Hợp tác" if label == 1 else "Không hợp tác"
            results.append({"text": sentence, "tone": tone})
        return results

    else:

        sentences = split_transcript_into_sentences(text)
        chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

        labels = []
        for chunk in chunks:
            chunk = clean_text(chunk)
            inputs = safe_tokenize(chunk, tokenizer)
            if inputs is None:
                continue
            label = predict(inputs)
            if label is None:
                continue
            labels.append(label)

        if not labels:
            return [{"text": text, "tone": "Error: all chunks invalid"}]

        final_label = max(set(labels), key=labels.count)
        tone = "Hợp tác" if final_label == 1 else "Không hợp tác"
        return [{"text": text, "tone": tone}]


def generate_response(text, label):
    prompt = f"""
    Bạn là nhân viên chăm sóc khách hàng của công ty tài chính, đang làm việc tại bộ phận thu hồi nợ.

    Thông tin khách hàng:
    - Cảm xúc hiện tại: {label}
    - Phát ngôn của khách hàng: "{text}"

    Yêu cầu:
    - Trả lời khách hàng bằng **tiếng Việt**, lịch sự, thân thiện, phù hợp hoàn cảnh.
    - Câu trả lời ngắn gọn, tự nhiên như người thật, không máy móc.
    - Nếu khách hợp tác, nhấn mạnh lịch hẹn thanh toán.
    - Nếu khách từ chối hoặc trì hoãn, hãy chọn cách xử lý phù hợp nhưng vẫn giữ thái độ thiện chí.

    Chỉ trả lời bằng phản hồi gửi cho khách hàng. Không giải thích thêm.

    Phản hồi:
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là nhân viên thu hồi nợ, hãy trả lời khách hàng bằng tiếng Việt, thân thiện, tự nhiên và thực tế."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def rule_based_response(text, label):
    text_lower = text.lower()
    if label == "Không hợp tác":
        if "không có tiền" in text_lower:
            return "Cám ơn bạn đã phản hồi. Chúng tôi sẽ liên hệ lại vào một ngày khác."
        if "cúp máy" in text_lower:
            return "Cám ơn bạn đã phản hồi. Chúng tôi sẽ liên hệ lại sau."
    return "Cám ơn bạn đã hợp tác. Chúng tôi sẽ tiếp tục theo dõi tình trạng của bạn."


def suggest_response(text, label, use_llm=True):
    if use_llm:
        return generate_response(text, label)
    else:
        return rule_based_response(text, label)


def cleanup_memory():
    gc.collect()
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    objgraph.show_growth(limit=10)


def evaluate_transcript(agent_transcript, sop_excel_file, method="embedding", use_rag=False, threshold=0.7):
    """
    Hàm đánh giá transcript dựa trên phương pháp được chọn: embedding, QA, hoặc RAG.
    """

    sheet_name = detect_sheet_from_text(agent_transcript)
    sop_items = extract_sop_items_from_excel(sop_excel_file, sheet_name=sheet_name)

    try:
        if method == "embedding":
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            eval_result = calculate_sop_compliance_by_sentences(
                agent_transcript, sop_items, model, threshold=threshold
            )
            return {"violations": eval_result}

        elif method == "qa":
            qa_llm, retriever, sop_data, combined_text = load_excel_rag_data(sop_excel_file)
            if not qa_llm:
                raise RuntimeError("Lỗi tải mô hình QA hoặc dữ liệu SOP.")

            response = qa_llm._call(prompt=agent_transcript)
            return {"answer": response}

        elif method == "rag":
            qa_llm, retriever, _, _ = load_excel_rag_data(sop_excel_file)

            if not qa_llm or not retriever:
                return {"error": "Lỗi tải mô hình RAG hoặc retriever từ dữ liệu SOP."}

            relevant_context = retriever.get_relevant_documents(agent_transcript)
            rag_context = "\n".join([doc.page_content for doc in relevant_context])
            response = qa_llm._call(prompt=agent_transcript, context=rag_context)
            return {"answer": response}


        else:
            raise ValueError(f"Phương pháp '{method}' không hợp lệ. Vui lòng chọn từ 'embedding', 'qa', hoặc 'rag'.")

    except Exception as e:
        return {"error": f"Lỗi khi đánh giá transcript: {str(e)}"}


def calculate_similarity(sentence, sop_item, model):
    embeddings = model.encode([sentence, sop_item], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def calculate_sop_compliance_by_sentences(transcript, sop_items, model, threshold=0.4):

    agent_sentences = split_transcript_into_sentences(transcript)

    flat_sop_items = []
    for section in sop_items:
        if 'items' in section and isinstance(section['items'], list):
            flat_sop_items.extend(section['items'])
        else:
            flat_sop_items.append(section)

    sop_compliance_results = []
    sop_violation_items = []

    for idx, sop_item in enumerate(flat_sop_items, 1):
        desc = sop_item.get('description', '')

        sub_sentences = split_sop_into_subsentences(desc)

        if not sub_sentences:
            sub_sentences = []


        def contains_proper_noun(sentence):
            words = re.findall(r'\b[A-ZÀ-Ỵ][a-zà-ỹ]+\b', sentence)
            filtered = [w for w in words if w.lower() not in {"em", "toi", "tôi", "anh", "chị", "chú", "cô", "bác"}]
            return len(filtered) > 0

        for sub_idx, sub_sentence in enumerate(sub_sentences, 1):
            matched = False
            status = "Chưa tuân thủ"
            lower_sub = sub_sentence.lower()

            if "xác định khách hàng" in lower_sub:
                if any(re.search(r"\b(chị|anh)\s+\w+", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "xác định tên của người thân" in lower_sub:
                if any(re.search(r"\b(chị|anh|vợ|chồng|cô|chú|bác)\s+\w+", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "xác định mối quan hệ" in lower_sub:
                if any(re.search(r"\b(chị|anh|vợ|chồng|cô|chú|bác)\s+\w+", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "cám ơn và chào khách hàng" in lower_sub:
                if any(re.search(r"cảm ơn", s.lower()) and re.search(r"chào", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "nhắn khách hàng gọi lại tổng đài" in lower_sub:
                if any(re.search(r"gọi", s.lower()) and re.search(r"tổng đài", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "đơn vị gọi đến" in lower_sub:
                  don_vi_keywords = [
                      r"\bhd\s*saigon\b",
                      r"\bbên\s+\w+",
                      r"\bcông\s+ty\b",
                  ]

                  nhac_no_keywords = [
                      r"\bnhắc\b",
                      r"\bthanh\s+toán\b",
                      r"\bkhoản\b",
                      r"\bnợ\b",
                      r"\btrễ\b"
                  ]

                  for s in agent_sentences:
                      s_lower = s.lower()
                      if any(re.search(kw, s_lower) for kw in don_vi_keywords) and any(re.search(kw, s_lower) for kw in nhac_no_keywords):
                          matched = True

                          status = "Đã tuân thủ"

            elif "đơn vị gọi đến" in lower_sub and "giới thiệu tên" in lower_sub:
                for s in agent_sentences:
                    s_lower = s.lower()
                    cond1 = contains_proper_noun(s) and "bên" in s_lower
                    cond2 = ("phòng công nợ" in s_lower or "công ty tài chính" in s_lower or
                             re.search(r"\bh\s*d\b|\bhd\b", s_lower) or "sài gòn" in s_lower or
                             "hcm" in s_lower or "chủ trả góp" in s_lower or "chào" in s_lower)
                    if cond1 and cond2:
                        matched = True
                        status = "Đã tuân thủ"

            elif "xin phép trao đổi" in lower_sub:
                if any(re.search(r"xin phép", s.lower()) and re.search(r"cho em hỏi", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "số ngày quá hạn" in lower_sub:
                if any(re.search(r"trễ", s.lower()) for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"

            elif "số tiền trễ hạn" in lower_sub:
                for s in agent_sentences:
                    s_lower = s.lower()

                    match = re.search(r"trễ\s+\d+\s+ngày.*(\d+[.,]?\d*\s*(ngàn|triệu|k|đ)?)", s_lower)
                    match_alt = re.search(r"trễ\s+\d+\s+ngày.*\d{1,3}([.,]\d{3})+", s_lower)

                    if match or match_alt:
                        matched = True
                        status = "Đã tuân thủ"

            elif "ghi nhận" in lower_sub:
                matched = True
                status = "Đã tuân thủ"

            elif "giọng nói" in lower_sub:
                matched = True
                status = "Đã tuân thủ"

            elif "ngôn ngữ" in lower_sub:
                matched = True
                status = "Đã tuân thủ"


            elif "không cúp máy ngang" in lower_sub:
                matched = True
                status = "Đã tuân thủ"

            else:
                if any(calculate_similarity(s, sub_sentence, model) >= threshold for s in agent_sentences):
                    matched = True
                    status = "Đã tuân thủ"


            score_val = sop_item.get("score", None)
            try:
                if score_val is None or score_val == "":
                    score_int = 0
                elif isinstance(score_val, (int, float)):
                    score_int = int(round(score_val))
                else:
                    cleaned_val = str(score_val).strip().lower()
                    if cleaned_val in ["", "nan", "none", "null"]:
                        score_int = 0
                    else:
                        score_int = int(round(float(cleaned_val)))
            except:
                score_int = 0

            stt_sub = f"{idx}.{sub_idx}"

            result_item = {
                "STT": stt_sub,
                "Tiêu chí": sub_sentence,
                "Trạng thái": status,
                "Điểm": score_int
            }

            sop_compliance_results.append(result_item)
            if status == "Chưa tuân thủ":
                sop_violation_items.append(result_item)

    processed_results = []
    for r in sop_compliance_results:
        if isinstance(r, dict):
            processed_results.append({
                "STT": str(r.get("STT", "?")) or "?",
                "Tiêu chí": str(r.get("Tiêu chí", "")),
                "Trạng thái": r.get("Trạng thái", "Không xác định"),
                "Điểm": r.get("Điểm", "")
            })

    valid_criteria = [
        item for item in processed_results
        if item.get("STT") != "" and item.get("Trạng thái") in ["Đã tuân thủ", "Chưa tuân thủ"]
    ]

    non_compliance_score = 0

    for item in valid_criteria:
        status = item.get("Trạng thái")
        raw_score = item.get("Điểm", 0)

        try:
            score = float(raw_score) if raw_score != "" else 0
        except ValueError:
            score = 0

        if status == "Chưa tuân thủ":
            non_compliance_score += score


    sop_compliance_rate = 100 - non_compliance_score

    formatted_violations = "\n".join(
        f"STT: {item['STT']} - Tiêu chí: {item['Tiêu chí']} - Điểm: {item['Điểm']}" for item in sop_violation_items
    )

    return processed_results, sop_compliance_rate, formatted_violations



def evaluate_sop_compliance(agent_transcript, sop_excel_file, model=None, threshold=0.3):

    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if isinstance(sop_excel_file, bytes):
        sop_excel_file = io.BytesIO(sop_excel_file)

    try:
        sheet_name = detect_sheet_from_text(agent_transcript)

        sop_items = extract_sop_items_from_excel(sop_excel_file, sheet_name=sheet_name)

        sop_results, sop_rate, sop_violations = calculate_sop_compliance_by_sentences(
            agent_transcript,
            sop_items,
            model,
            threshold=threshold
        )

        expected_keys = ["STT", "Tiêu chí", "Trạng thái", "Điểm"]
        safe_results = []

        for result in sop_results:
            if isinstance(result, dict):
                clean_result = {k: result.get(k, "") for k in expected_keys}

                clean_result["STT"] = str(clean_result["STT"]).strip() or "?"

                try:
                    clean_result["Điểm"] = int(float(clean_result["Điểm"]))
                except Exception:
                    clean_result["Điểm"] = 0

                if clean_result["Trạng thái"] not in ["Đã tuân thủ", "Chưa tuân thủ"]:
                    clean_result["Trạng thái"] = "Không xác định"

                clean_result["Tiêu chí"] = str(clean_result["Tiêu chí"])

                safe_results.append(clean_result)

        return safe_results, sop_rate, sop_violations

    except Exception as e:
        return [{
            "STT": "?",
            "Tiêu chí": f"Lỗi khi đánh giá mức độ tuân thủ SOP: {e}",
            "Trạng thái": "Lỗi",
            "Điểm": ""
        }], 0.0, []



def split_violation_text(violation_text):
    if not isinstance(violation_text, str):
        return [{"STT": "?", "Tiêu chí": str(violation_text)}]

    lines = violation_text.strip().split("\n")
    violations = []

    for line in lines:
        clean_line = line.strip()
        if clean_line:
            violations.append({"STT": "?", "Tiêu chí": clean_line})
    return violations


def evaluate_combined_transcript_and_compliance(agent_transcript, sop_excel_file, method=None, threshold=0.7):
    """
    Đánh giá transcript bằng mô hình RAG và tính toán độ tuân thủ SOP.
    """
    selected_method = method or "rag"
    eval_result = {}

    sop_results, sop_rate, sop_violations = evaluate_sop_compliance(
        agent_transcript, sop_excel_file, threshold=threshold
    )

    if not isinstance(sop_violations, list):
        sop_violations = [{"STT": "?", "Tiêu chí": str(sop_violations)}]
    else:
        sop_violations = [
            {"STT": "?", "Tiêu chí": str(v)} if not isinstance(v, dict) else v
            for v in sop_violations
        ]


    if len(sop_violations) == 1 and isinstance(sop_violations[0], dict):
        raw_text = sop_violations[0].get("Tiêu chí", "")
        if "\n" in raw_text:
            sop_violations = split_violation_text(raw_text)


    eval_result["sop_compliance_results"] = sop_results
    eval_result["compliance_rate"] = sop_rate
    eval_result["violations"] = sop_violations

    if selected_method == "rag":
        try:
            qa_llm, retriever, sop_data, combined_text = load_excel_rag_data(sop_excel_file)
            if not qa_llm or not retriever:
                eval_result["rag_explanations"] = "Không thể tải mô hình hoặc dữ liệu RAG."
                eval_result["selected_method"] = selected_method
                return eval_result

            rag_explanations = []

            for violation in sop_violations:
                try:
                    if isinstance(violation, dict):
                        sop_criterion = violation.get("Tiêu chí", str(violation))
                    else:
                        sop_criterion = str(violation)

                    relevant_context = retriever.get_relevant_documents(sop_criterion)
                    rag_context = "\n".join([doc.page_content for doc in relevant_context])
                    rag_response = qa_llm._call(prompt=sop_criterion, context=rag_context)

                    rag_explanations.append({
                        "Tiêu chí": sop_criterion,
                        "Giải thích từ RAG": rag_response
                    })

                except Exception as e:
                    rag_explanations.append({
                        "Tiêu chí": str(violation),
                        "Giải thích từ RAG": f"Lỗi: {e}"
                    })

            eval_result["rag_explanations"] = rag_explanations

        except Exception as e:
            eval_result["rag_explanations"] = f"Lỗi khi sử dụng RAG: {e}"

    eval_result["selected_method"] = selected_method
    return eval_result


def process_files(uploaded_excel_file, uploaded_audio_file):
    uploaded_excel_file.seek(0)
    qa_llm, retriever, sop_data, combined_text = load_excel_rag_data(uploaded_excel_file)

    transcripts_by_file = {}
    detected_sheets_by_file = {}

    uploaded_audio_file.seek(0)

    if uploaded_audio_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_audio_file) as z:
            for file_name in z.namelist():
                if file_name.lower().endswith(".wav"):
                    with z.open(file_name) as wav_file:
                        file_like = BytesIO(wav_file.read())
                        file_like.seek(0)
                        transcript = transcribe_audio(file_like)
                        transcripts_by_file[file_name] = transcript
                        detected_sheets_by_file[file_name] = detect_sheet_from_text(transcript)

    elif uploaded_audio_file.name.lower().endswith(".wav"):
        transcript = transcribe_audio(uploaded_audio_file)
        transcripts_by_file[uploaded_audio_file.name] = transcript
        detected_sheets_by_file[uploaded_audio_file.name] = detect_sheet_from_text(transcript)

    else:
        raise ValueError("Định dạng tệp âm thanh không hợp lệ. Chỉ hỗ trợ .zip hoặc .wav")

    return qa_llm, retriever, sop_data, transcripts_by_file, detected_sheets_by_file


def process_files_from_zip_transcripts(uploaded_excel_file, uploaded_zip_file):
    uploaded_excel_file.seek(0)
    qa_llm, retriever, sop_data, combined_text = load_excel_rag_data(uploaded_excel_file)

    transcripts_by_file = {}
    detected_sheets_by_file = {}

    uploaded_zip_file.seek(0)

    if uploaded_zip_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_zip_file) as z:
            for file_name in z.namelist():
                if file_name.lower().endswith(".txt"):
                    with z.open(file_name) as txt_file:
                        transcript = txt_file.read().decode("utf-8")
                        transcripts_by_file[file_name] = transcript
                        detected_sheets_by_file[file_name] = detect_sheet_from_text(transcript)
    else:
        raise ValueError("Tệp không phải .zip chứa transcript dạng .txt")

    return qa_llm, retriever, sop_data, transcripts_by_file, detected_sheets_by_file



def export_combined_sheet_per_file(df_all_concat, criteria_orders_by_file):
    if df_all_concat.empty:
        return None

    meta_cols_head = ["Tên file audio", "Loại cuộc gọi"]
    meta_cols_tail = [
        "Tỷ lệ tuân thủ tổng thể",
        "Chi tiết lỗi đánh giá - đơn vị",
        "Tỷ lệ phản hồi tích cực của KH",
        "Ghi chú - đơn vị",
        "Phản hồi gợi ý"
    ]

    rows = []
    all_columns = set()

    for _, row in df_all_concat.iterrows():
        file_name = row["Tên file audio"]
        criteria_order = criteria_orders_by_file.get(file_name, [])

        all_cols_ordered = meta_cols_head + criteria_order + meta_cols_tail
        ordered_row = pd.Series({col: row.get(col, "") for col in all_cols_ordered})
        rows.append(ordered_row)
        all_columns.update(all_cols_ordered)

    df_final = pd.DataFrame(rows)

    final_columns = meta_cols_head + sorted(list(all_columns - set(meta_cols_head) - set(meta_cols_tail))) + meta_cols_tail
    df_final = df_final.reindex(columns=final_columns)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_final.to_excel(writer, sheet_name="Tong_hop_cuoc_goi", index=False)
    output.seek(0)
    return output


st.title("Đánh giá Cuộc Gọi - AI Bot")

def main():
    uploaded_excel_file = st.file_uploader("Tải lên tệp Excel", type="xlsx")
    uploaded_audio_file = st.file_uploader("Tải lên tệp âm thanh (.zip hoặc .wav)", type=["zip", "wav"])

    if uploaded_excel_file and uploaded_audio_file:
        st.success("Tải tệp thành công!")

        if st.button("Đánh giá"):
            try:
                qa_chain, retriever, sop_data, transcripts_by_file, detected_sheets_by_file = process_files_from_zip_transcripts(
                    uploaded_excel_file, uploaded_audio_file
                )
            except Exception as e:
                st.error(f"Lỗi khi xử lý tệp: {e}")
                return

            df_all = []
            criteria_orders_by_file = {}

            for file_name, transcript in transcripts_by_file.items():
                st.subheader(f"Tệp âm thanh: {file_name}")
                st.write(transcript)

                analysis_result = analyze_call_transcript(transcript)
                tone_chunks = analysis_result["tone_chunks"]
                # customer_label = classify_tone(transcript, chunk_size=None)
                tone_results = classify_tone(transcript, chunk_size=None)


                if isinstance(tone_results, list) and isinstance(tone_results[0], dict):
                    tone_labels = [r["tone"] for r in tone_results if "tone" in r]
                    customer_label = max(set(tone_labels), key=tone_labels.count) if tone_labels else "Không xác định"
                else:
                    customer_label = "Không xác định"


                notes_parts = [
                    f"Cảm xúc của khách hàng: {customer_label}",
                    f"Ý định của khách hàng: {analysis_result['intent']}",
                    "Tổng hợp cảm xúc trong câu của khách hàng:"
                ]

                for tone, count in tone_chunks["tone_summary"].items():
                    notes_parts.append(f"  - {tone}: {count} câu")

                notes_parts.append("Các đoạn nổi bật:")
                for chunk in tone_chunks["important_chunks"]:
                    notes_parts.append(f"  - > \"{chunk['text']}\"\n    → **{chunk['tone']}**")

                note_text = "\n".join(notes_parts)

                try:
                    results = evaluate_combined_transcript_and_compliance(
                        transcript,
                        uploaded_excel_file,
                        method="rag",
                        threshold=0.6
                    )
                    compliance_rate = results.get('compliance_rate', 0.0)
                    sop_results = results.get('sop_compliance_results', [])
                    sop_violations = results.get('violations', [])

                    if sop_results:
                        df_sop_results = pd.DataFrame(sop_results)
                        df_sop_results = df_sop_results[df_sop_results["Trạng thái"].astype(str).str.strip() != ""]

                        df_sop_results["Trạng thái"] = df_sop_results["Trạng thái"].apply(
                            lambda x: "Y" if str(x).strip().lower() == "đã tuân thủ" else "N"
                        )

                        sheet_name_detected = detected_sheets_by_file.get(file_name, "").strip()
                        if "NT" in sheet_name_detected:
                            call_type = "NT"
                        elif "KH" in sheet_name_detected:
                            call_type = "KH"
                        else:
                            call_type = "Unknown"

                        df_sop_results["Tiêu chí prefixed"] = df_sop_results["Tiêu chí"] + f"_{call_type}"
                        criteria_order_prefixed = df_sop_results["Tiêu chí prefixed"].tolist()
                        criteria_orders_by_file[file_name] = criteria_order_prefixed

                        df_pivot = df_sop_results.pivot_table(
                            index=[],
                            columns="Tiêu chí prefixed",
                            values="Trạng thái",
                            aggfunc='first'
                        ).reset_index(drop=True)
                        df_pivot = df_pivot.reindex(columns=criteria_order_prefixed).fillna("")

                        suggestion = suggest_response(transcript, customer_label, use_llm=True)

                        metadata = {
                            "Tên file audio": file_name,
                            "Loại cuộc gọi": call_type,
                            "Tỷ lệ tuân thủ tổng thể": f"{compliance_rate:.2f}%",
                            "Chi tiết lỗi đánh giá - đơn vị": "\n".join(
                                f"{v.get('Tiêu chí', '')}" for v in sop_violations if isinstance(v, dict)
                            ),
                            "Tỷ lệ phản hồi tích cực của KH": f"{analysis_result['collaboration_rate']}%",
                            "Ghi chú - đơn vị": note_text,
                            "Phản hồi gợi ý": suggestion
                        }

                        df_info = pd.DataFrame([metadata])

                        df_criteria_full = pd.DataFrame(columns=criteria_order_prefixed)
                        for crit in criteria_order_prefixed:
                            df_criteria_full[crit] = df_pivot[crit] if crit in df_pivot.columns else ""

                        cols_head = ["Tên file audio", "Loại cuộc gọi"]
                        df_metadata_head = df_info[cols_head]

                        cols_tail = [col for col in df_info.columns if col not in cols_head]
                        df_metadata_tail = df_info[cols_tail]

                        df_concat = pd.concat([df_metadata_head, df_criteria_full, df_metadata_tail], axis=1)

                        df_all.append(df_concat)

                except Exception as e:
                    st.error(f"Lỗi khi đánh giá compliance: {e}")


            if df_all:
                df_all_concat = pd.concat(df_all, axis=0, ignore_index=True)

                df_kh_all = df_all_concat[df_all_concat["Loại cuộc gọi"] == "KH"]
                df_nt_all = df_all_concat[df_all_concat["Loại cuộc gọi"] == "NT"]


                excel_file = export_combined_sheet_per_file(df_all_concat, criteria_orders_by_file)

            st.download_button(
                label="Tải file tổng hợp Excel",
                data=excel_file,
                file_name="AI_QA_GRACE.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            

        cleanup_memory()


if __name__ == "__main__":
    main()
