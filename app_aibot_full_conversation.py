
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
from collections import defaultdict, Counter
from unidecode import unidecode
nest_asyncio.apply()



if sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())



def transcribe_audio(uploaded_file):
    transcription = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=uploaded_file,
        language="vi"
    )
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
        "Xin gọi lại": [
            r"đang bận", r"gọi giờ khác", r"đang họp"
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
    if isinstance(file_path, BytesIO):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
    elif isinstance(file_path, pd.DataFrame):
        df = file_path
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    required_columns = ['Mã tiêu chí', 'Tên tiêu chí đánh giá', 'Điểm', 'Hướng dẫn thực hiện', 'Hướng dẫn đánh giá']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    df = df[required_columns]
    df[['Tên tiêu chí đánh giá', 'Hướng dẫn thực hiện']] = df[['Tên tiêu chí đánh giá', 'Hướng dẫn thực hiện']].ffill()
    df.fillna("", inplace=True)

    sop_items = []
    current_section = None

    for _, row in df.iterrows():
        code = str(row['Mã tiêu chí']).strip()
        title = str(row['Tên tiêu chí đánh giá']).strip()
        score = row['Điểm']
        implementation = str(row['Hướng dẫn thực hiện']).strip()
        evaluation_guide = str(row['Hướng dẫn đánh giá']).strip()

        if (
            not code and not title
        ):
            continue  

        if code.isupper() and code:
            merged_text = " - ".join(filter(None, [title, implementation, evaluation_guide]))
            sop_items.append({
                "section_header": None,
                "full_text": f"{code}",
                "score": score, 
                "implementation": merged_text, 
                "evaluation_guide": "",
                "is_section_header": True
            })
            continue


    return sop_items


def split_into_sentences(text):
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    merged = []
    buffer = ""
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= 3:
            buffer += " " + sentence
        else:
            if buffer:
                merged.append((buffer + " " + sentence).strip())
                buffer = ""
            else:
                merged.append(sentence)
    if buffer:
        merged.append(buffer.strip())
    return merged


IGNORE_KEYWORDS = [
    "alo", "chào", "em gọi", "cho em hỏi", "không ạ", "bên em", "đơn vị", "công ty", "em là", "gọi cho chị", "từ bên", "liên kết",
    "chậm nhất", "thanh toán", "hồ sơ", "ngân hàng", "báo cáo", "giùm em", "hả", "xin phép gọi lại sau", "nói với", "nhờ chị",
    "không biết là", "báo cho chị", "chuyển luôn cho em", "Đúng rồi", "liên lạc lại sau", "đúng không", "chưa chị", "chị tính",
    "mình đóng", "em báo", "nhắc", "xử lý", "thu", "vậy ạ", "dạ không đổi số", "giúp nha"
]


def is_greeting_or_intro(sentence):
    sentence_lower = sentence.lower()
    return any(kw in sentence_lower for kw in IGNORE_KEYWORDS)


def calculate_similarity(sentence, sop_item, model):
    embeddings = model.encode([sentence, sop_item], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def calculate_sop_compliance_by_sentences(transcript, sop_items, model, threshold=0.4):
    agent_sentences = split_into_sentences(transcript)

    compliant_sentences = sum(
        any(calculate_similarity(sentence, sop_item['full_text'], model) >= threshold for sop_item in sop_items)
        for sentence in agent_sentences
    )

    sentence_compliance_percentage = (
        (compliant_sentences / len(agent_sentences)) * 100 if agent_sentences else 0
    )

    sop_compliance_results = []
    sop_violation_items = []


    for idx, sop_item in enumerate(sop_items, 1):
        matched = False
        status = "Chưa tuân thủ"

        lower_item = sop_item['full_text'].lower()


        if sop_item.get("is_section_header"):
            sop_compliance_results.append({
                "STT": "",
                "Tiêu chí": sop_item["full_text"].upper(),
                "Trạng thái": "",
                "Điểm": ""
            })
            continue

        if "xác định khách hàng" in lower_item:
            if any(re.search(r"\b(chị|anh)\s+\w+", s.lower()) for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"

        elif "xác định người thân" in lower_item:
            if any(re.search(r"\b(chị|anh|cô|chú|bác)\s+\w+", s.lower()) for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"

        elif re.search(r"\b(chị|anh)\s+\w+", lower_item):
            matched = True
            status = "Đã tuân thủ"

        elif "cám ơn và chào khách hàng" in lower_item:
            if any(re.search(r"cảm ơn", s.lower()) and re.search(r"chào", s.lower()) for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"

        elif "lời nhắn" in lower_item:
            if any(re.search(r"lời nhắn", s.lower()) for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"

        elif "đơn vị gọi đến" in lower_item or "giới thiệu tên" in lower_item:
            for s in agent_sentences:
                s_lower = s.lower()
                if (
                    ("em bên" in s_lower in s_lower or "bên" in s_lower) or
                    (
                        "phòng công nợ" in s_lower or
                        "công ty tài chính" in s_lower or
                        re.search(r"\bh\s*d\b|\bhd\b", s_lower) or
                        "sài gòn" in s_lower or
                        "hcm" in s_lower or
                        "chủ trả góp" in s_lower
                    ) and
                    (
                        "chào" in s_lower or
                        "xin phép trao đổi" in s_lower or
                        "xin phép nói chuyện" in s_lower or
                        "alo" in s_lower or
                        "cho em hỏi" in s_lower
                    )
                ):
                    matched = True
                    status = "Đã tuân thủ"


        elif "Ghi nhận kết quả cuộc gọi" in lower_item:
            matched = True
            status = "Đã tuân thủ"


        elif "giọng nói" in lower_item:
            matched = True
            status = "Đã tuân thủ"

        elif "ngôn ngữ" in lower_item:
            matched = True
            status = "Đã tuân thủ"

        elif "hotline" in lower_item:
            if any("1900558854" in s for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"
        else:
            if any(calculate_similarity(s, sop_item['full_text'], model) >= threshold for s in agent_sentences):
                matched = True
                status = "Đã tuân thủ"



        score_val = sop_item.get("score")
        if pd.notna(score_val) and score_val != "":
            score_int = int(round(score_val))
        else:
            score_int = 0

        sop_compliance_results.append({
            "STT": idx,
            "Tiêu chí": sop_item['full_text'],
            "Trạng thái": status,
            "Điểm": score_int
        })

        if status == "Chưa tuân thủ":
            sop_violation_items.append({
                "STT": idx,
                "Tiêu chí": sop_item['full_text'],
                "Điểm": score_int
            })

    valid_criteria = [item for item in sop_compliance_results if item["STT"] != ""]
    complied_criteria = [item for item in valid_criteria if item["Trạng thái"] == "Đã tuân thủ"]

    sop_compliance_rate = (
        len(complied_criteria) / len(valid_criteria) * 100
        if valid_criteria else 0
    )

    formatted_violations = "\n".join(
        f"STT: {item['STT']} - Tiêu chí: {item['Tiêu chí']} - Điểm: {item['Điểm']}" for item in sop_violation_items
    )

    return sop_compliance_results, sop_compliance_rate, sentence_compliance_percentage, formatted_violations



def evaluate_sop_compliance(agent_transcript, sop_excel_file, model=None, threshold=0.3):

    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if isinstance(sop_excel_file, bytes):
        sop_excel_file = io.BytesIO(sop_excel_file)

    sheet_name = detect_sheet_from_text(agent_transcript)

    sop_items = extract_sop_items_from_excel(sop_excel_file, sheet_name=sheet_name)

    try:
        sop_results, sop_rate, sentence_rate, sop_violations = calculate_sop_compliance_by_sentences(
            agent_transcript,
            sop_items,
            model,
            threshold=threshold
        )

        if sop_violations is None:
            sop_violations = []
        return sop_results, sop_rate, sentence_rate, sop_violations
    except Exception as e:
        return [{
            "STT": "?",
            "Tiêu chí": f"Lỗi khi tính SOP: {e}",
            "Trạng thái": "Lỗi",
            "Điểm": ""
        }], 0.0, 0.0, []



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


    entities_dict = {}
    total_sentences = 0
    cooperative_sentences = 0
    tone_counter = Counter()
    tone_chunks_result = []

    for sentence in merged_sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < min_sentence_length or is_greeting_or_intro(sentence):
            continue

        total_sentences += 1
        try:
            tones = classify_tone(sentence)
        except Exception as e:
            print(f"Error classifying tone for sentence: {e}")
            tones = [{"text": sentence, "tone": "Error"}]

        if isinstance(tones, list):
            for tone in tones:
                tone_value = tone.get('tone', '') if isinstance(tone, dict) else tone
                tone_chunks_result.append({
                    "text": sentence,
                    "tone": tone_value
                })
                tone_counter[tone_value] += 1
                if tone_value == "Hợp tác":
                    cooperative_sentences += 1
        elif isinstance(tones, dict):
            tone_value = tones.get('tone', '')
            tone_chunks_result.append({
                "text": sentence,
                "tone": tone_value
            })
            tone_counter[tone_value] += 1
            if tone_value == "Hợp tác":
                cooperative_sentences += 1

        try:
            ner_results = ner_pipeline(sentence)
            for entity in ner_results:
                label = entity['entity_group']
                word = entity['word']
                entities_dict.setdefault(label, set()).add(word)
        except Exception as e:
            print(f"NER error: {e}")

    try:
        intent_result = detect_intent(text)
    except:
        intent_result = "Không xác định"

    collaboration_rate = (cooperative_sentences / total_sentences) * 100 if total_sentences > 0 else 0


    interaction_summary = "=== Đánh giá tương tác ===\n"
    if collaboration_rate < 50:
        interaction_summary += f"Tỷ lệ hợp tác thấp ({collaboration_rate:.2f}%). Cần chú ý các câu sau:\n"
        for i, item in enumerate(tone_chunks_result):
            if item["tone"] == "Không hợp tác":
                interaction_summary += f"{i+1}. {item['text']}\n"
    else:
        interaction_summary += f"Tỷ lệ hợp tác cao ({collaboration_rate:.2f}%). Những câu nổi bật:\n"
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
        sentences = split_into_sentences(text)
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

        sentences = split_into_sentences(text)
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
    - Câu trả lời ngắn gọn (1-2 câu), tự nhiên như người thật, không máy móc.
    - Nếu khách hợp tác, nhấn mạnh lịch hẹn thanh toán.
    - Nếu khách từ chối hoặc trì hoãn, hãy chọn cách xử lý phù hợp nhưng vẫn giữ thái độ thiện chí.

    Chỉ trả lời bằng phản hồi gửi cho khách hàng. Không giải thích thêm.

    Phản hồi:
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
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



def split_violation_text(violation_text):
    """
    Tách chuỗi tiêu chí dài thành danh sách dict {"STT": "?", "Tiêu chí": dòng cụ thể}
    """
    if not isinstance(violation_text, str):
        return [{"STT": "?", "Tiêu chí": str(violation_text)}]

    lines = violation_text.strip().split("\n")
    violations = []

    for line in lines:
        clean_line = line.strip()
        if clean_line:
            violations.append({"STT": "?", "Tiêu chí": clean_line})


def evaluate_combined_transcript_and_compliance(agent_transcript, sop_excel_file, method=None, threshold=0.7):
    """
    Đánh giá transcript bằng mô hình RAG và tính toán độ tuân thủ SOP.
    """
    selected_method = method or "rag"
    eval_result = {}

    try:
        sop_results, sop_rate, sentence_rate, sop_violations = evaluate_sop_compliance(
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
        eval_result["sentence_compliance_rate"] = sentence_rate
        eval_result["violations"] = sop_violations

    except Exception as e:
        eval_result["sop_compliance_results"] = "Lỗi khi đánh giá tuân thủ SOP."
        eval_result["violations"] = f"Lỗi: {e}"

        try:
            debug_data = {"agent_transcript": agent_transcript, "error": str(e)}
            debug_json = json.dumps(debug_data, ensure_ascii=False, indent=4)
            with st.expander("Chi tiết lỗi debug"):
                st.code(debug_json, language="json")
        except Exception as file_error:
            eval_result["file_save_error"] = f"Không thể ghi file debug: {file_error}"
            st.error(eval_result["file_save_error"])


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

            if rag_explanations:
                eval_result["rag_explanations"] = rag_explanations

        except Exception as e:
            eval_result["rag_explanations"] = f"Lỗi khi sử dụng RAG: {e}"

    eval_result["selected_method"] = selected_method
    return eval_result


# func process audio by batch
def process_files(uploaded_excel_file, uploaded_zip_audio):
    try:
        qa_llm, retriever, sop_data, combined_text = load_excel_rag_data(uploaded_excel_file)

        transcripts_by_file = {}
        detected_sheets_by_file = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(uploaded_zip_audio, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            audio_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        audio_files.append(os.path.join(root, file))

            for i, file_path in enumerate(audio_files, start=1):
                file_name = os.path.basename(file_path)

                try:

                    with open(file_path, "rb") as audio_file:
                        transcript = transcribe_audio(audio_file)
                    transcripts_by_file[file_name] = transcript


                    detected_sheet = detect_sheet_from_text(transcript)
                    detected_sheets_by_file[file_name] = detected_sheet

                except Exception as e:
                    print(f"Lỗi khi xử lý {file_name}: {e}")
                    transcripts_by_file[file_name] = ""
                    detected_sheets_by_file[file_name] = ""

        return qa_llm, retriever, sop_data, transcripts_by_file, detected_sheets_by_file

    except Exception as e:
        print(f"Lỗi khi xử lý batch file: {e}")
        return None, None, None, {}, {}


# func process each file audio
def process_audio_file(file_path):
    try:
        transcript = transcribe_audio(file_path)

        detected_sheet = detect_sheet_from_text(transcript)

        file_name = os.path.basename(file_path)
        return file_name, transcript, detected_sheet
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return os.path.basename(file_path), "", ""


st.title("Đánh giá Cuộc Gọi - AI Bot")

def main():
    uploaded_excel_file = st.file_uploader("Tải lên tệp Excel", type="xlsx")
    uploaded_audio_file = st.file_uploader("Tải lên tệp âm thanh (ZIP chứa các tệp .wav)", type=["zip"])

    if uploaded_excel_file and uploaded_audio_file:
        st.success("Tải tệp thành công!")

        if st.button("Đánh giá"):
            with st.spinner("Đang xử lý..."):
                try:
                    qa_chain, retriever, sop_data, transcripts_by_file, detected_sheets_by_file = process_files(
                        uploaded_excel_file, uploaded_audio_file
                    )

                except Exception as e:
                    st.error(f"Lỗi khi xử lý tệp: {e}")
                    return

                st.subheader("Văn bản thu được từ các tệp âm thanh:")
                for file_name, transcript in transcripts_by_file.items():
                    st.write(f"**Tệp âm thanh**: {file_name}")
                    st.write(transcript)

                    analysis_result = analyze_call_transcript(transcript)
                    tone_chunks = analysis_result["tone_chunks"]
                    customer_label = classify_tone(transcript, chunk_size=None)

                    st.subheader("Kết quả phân tích:")
                    st.write(f"Cảm xúc của khách hàng: {customer_label}")
                    st.write(f"Thực thể được nhận diện: {analysis_result['named_entities']}")
                    st.write(f"Ý định của khách hàng: {analysis_result['intent']}")
                    st.write(f"Tỷ lệ hợp tác: {analysis_result['collaboration_rate']}%")
                    st.text(analysis_result["interaction_summary"])

                    st.header("Tổng hợp cảm xúc trong câu của khách hàng")
                    for tone, count in tone_chunks["tone_summary"].items():
                        st.write(f"- {tone}: {count} câu")

                    st.subheader("Các đoạn nổi bật:")
                    for chunk in tone_chunks["important_chunks"]:
                        st.markdown(f"> \"{chunk['text']}\"\n→ **{chunk['tone']}**")


                    st.subheader("Đánh giá mức độ tuân thủ SOP:")
                    try:
                        results = evaluate_combined_transcript_and_compliance(
                            transcript,
                            uploaded_excel_file,
                            method="rag",
                            threshold=0.4
                        )

                        st.subheader("Tỷ lệ tuân thủ tổng thể:")
                        st.markdown(f"- **{results['compliance_rate']:.2f}%**")

                        st.subheader("Tỷ lệ tuân thủ theo từng câu:")
                        st.markdown(f"- **{results['sentence_compliance_rate']:.2f}%**")

                        st.subheader("Chi tiết từng tiêu chí:")
                        df_sop_results = pd.DataFrame(results['sop_compliance_results'])
                        df_sop_results = df_sop_results[["Tiêu chí", "Trạng thái", "Điểm"]]
                        df_sop_results = df_sop_results.drop_duplicates(subset=["Tiêu chí"])
                        df_sop_results = df_sop_results.reset_index(drop=True)

                        st.table(df_sop_results)

                        df_violations = df_sop_results[df_sop_results["Trạng thái"] != "Đã tuân thủ"]

                        if not df_violations.empty:
                            df_violations = df_violations[["Tiêu chí", "Điểm"]]
                            df_violations["Điểm"] = df_violations["Điểm"].astype(int)
                            df_violations = df_violations.reset_index(drop=True)

                            st.subheader("Các tiêu chí chưa tuân thủ:")
                            st.table(df_violations)
                        else:
                            st.success("Nhân viên đã tuân thủ đầy đủ các tiêu chí SOP!")

                        st.subheader("Phản hồi gợi ý:")
                        suggestion = suggest_response(transcript, customer_label, use_llm=True)
                        st.write(suggestion)

                    except Exception as e:
                        st.error(f"Lỗi khi đánh giá mức độ tuân thủ SOP cho tệp {file_name}: {e}")

                cleanup_memory()

if __name__ == "__main__":
    main()
