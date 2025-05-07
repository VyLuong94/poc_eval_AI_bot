
import warnings
warnings.filterwarnings("ignore")
import nest_asyncio # no running event loop
import asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import os
import re
from collections import Counter
import openai
import httpx
import torch
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

# --- WHISPER FUNCTION ---

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


# --- INTENT DETECTION VIA REGEX ---


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
            r"gọi lại sau", r"đang bận", r"gọi giờ khác", r"đang họp"
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


# --- SOP COMPLIANCE FUNCTIONS ---

def extract_sop_items(sop_text):
    sop_items = re.findall(r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)", sop_text.strip(), re.DOTALL)
    return [item.strip().replace('\n', ' ') for item in sop_items]

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Tính toán sự tương đồng giữa 2 câu sử dụng cosine similarity
def calculate_similarity(sentence, sop_item, model):
    embeddings = model.encode([sentence, sop_item], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Tính toán tỷ lệ tuân thủ SOP theo từng câu
def calculate_sop_compliance_by_sentences(transcript, combined_text, model, threshold=0.8):
    sop_items = extract_sop_items(combined_text)
    agent_sentences = split_into_sentences(transcript)
    # Đánh giá sự tuân thủ của từng câu
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    compliant_sentences = 0
    for sentence in agent_sentences:
        if any(calculate_similarity(sentence, sop_item, model) >= threshold for sop_item in sop_items):
            compliant_sentences += 1

    sentence_compliance_percentage = (compliant_sentences / len(agent_sentences)) * 100 if len(agent_sentences) > 0 else 0

    # Đánh giá sự tuân thủ của từng mục SOP
    sop_compliance_results = []
    sop_violation_items = []
    for idx, sop_item in enumerate(sop_items, 1):
        matched = any(calculate_similarity(sentence, sop_item, model) >= threshold for sentence in agent_sentences)
        status = "Đã tuân thủ" if matched else "Chưa tuân thủ"
        sop_compliance_results.append((idx, sop_item, status))
        if status == "Chưa tuân thủ":
            sop_violation_items.append((idx, sop_item))

    sop_compliance_rate = sum(1 for _, _, status in sop_compliance_results if status == "Đã tuân thủ") / len(sop_compliance_results) * 100 if len(sop_compliance_results) > 0 else 0

    return sop_compliance_results, sop_compliance_rate, sentence_compliance_percentage, sop_violation_items


def evaluate_sop_compliance(agent_transcript, sop_data, model, threshold=0.8):
    # Xác định loại cuộc gọi dựa trên nội dung transcript
    selected_sheet = detect_sheet_from_text(agent_transcript)

    # Lấy danh sách các mục SOP phù hợp
    sop_items_text = "\n".join(sop_data[selected_sheet])

    # Tính toán tuân thủ SOP
    return calculate_sop_compliance_by_sentences(
        agent_transcript,
        sop_items_text,
        model,
        threshold=threshold
    )


def detect_sheet_from_text(agent_text):
    keywords_relative = ['vợ', 'chồng', 'con', 'người thân', 'gia đình', 'người nhà']
    if any(keyword in agent_text.lower() for keyword in keywords_relative):
        return 'Tiêu chí giám sát cuộc gọi NT'
    else:
        return 'Tiêu chí giám sát cuộc gọi KH'


# --- ANALYSIS FUNCTION ---

# Use HuggingFace NER to assign speakers
@st.cache_resource
def load_ner_pipeline():
    """Load NER pipeline with Hugging Face token."""
    model_name = "NlpHUST/ner-vietnamese-electra-base"
    hf_api_token = st.secrets["huggingface"]["token"]

    try:
        # Load model and tokenizer with token
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_token)
        model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=hf_api_token)

        # Create NER pipeline
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


# Main analysis function
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

def analyze_call_transcript(text, max_chunk_length=128, min_sentence_length=5, client=None):
    raw_sentences = re.split(r'(?<=[.!?]) +', text)
    merged_sentences = merge_short_sentences(raw_sentences)

    chunks = []
    current_chunk = ""

    for sentence in merged_sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < min_sentence_length and current_chunk:
            current_chunk += sentence + " "
        else:
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Text split into {len(chunks)} chunks")

    entities_dict = {}
    total_sentences = 0
    cooperative_sentences = 0
    tone_counter = Counter()
    tone_chunks_result = []

    for chunk in chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if sentence:
                total_sentences += 1
                try:
                    tones = classify_tone(sentence)
                except Exception as e:
                    print(f"Error classifying tone for sentence: {e}")
                    tones = [{"text": sentence, "tone": "Error"}]

                # Phân loại tone của câu
                if isinstance(tones, list):
                    for tone in tones:
                        if isinstance(tone, dict):
                            tone = tone.get('tone', '')
                        tone_chunks_result.append({
                            "text": sentence,
                            "tone": tone
                        })
                        tone_counter[tone] += 1
                        print(f"Sentence: {sentence} - Tone: {tone}")
                        if tone == "Hợp tác":
                            cooperative_sentences += 1
                elif isinstance(tones, dict):
                    tone = tones.get('tone', '')
                    if tone:
                        tone_chunks_result.append({
                            "text": sentence,
                            "tone": tone
                        })
                        tone_counter[tone] += 1
                        print(f"Sentence: {sentence} - Tone: {tone}")
                        if tone == "Hợp tác":
                            cooperative_sentences += 1


        ner_results = ner_pipeline(chunk)
        for entity in ner_results:
            label = entity['entity_group']
            word = entity['word']
            entities_dict.setdefault(label, set()).add(word)

    collaboration_rate = (cooperative_sentences / total_sentences) * 100 if total_sentences > 0 else 0
    important_chunks = [chunk for chunk in tone_chunks_result if chunk["tone"] in ["Hợp tác", "Không hợp tác"]]
    intent_result = detect_intent(text)

    interaction_summary = "=== Đánh giá tương tác theo câu ===\n"
    if collaboration_rate < 50:
        interaction_summary += f"Tỷ lệ hợp tác thấp ({collaboration_rate:.2f}%). Những câu không hợp tác cần chú ý:\n"
        for i, item in enumerate(tone_chunks_result):
            if item["tone"] == "Không hợp tác":
                interaction_summary += f"{i+1}. {item['text']}\n"
    else:
        interaction_summary += f"Tỷ lệ hợp tác cao ({collaboration_rate:.2f}%). Những câu hợp tác đóng góp tích cực:\n"
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
            "important_chunks": important_chunks
        }
    }


# --- RESPONSE SUGGESTION FUNCTIONS ---

# Cache models once
@st.cache_resource
def load_model():
    """Load tone classification model and tokenizer."""
    model_name = "vyluong/tone-classification-model"
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


# Load and process Excel data for RAG
@st.cache_resource
def load_excel_rag_data(uploaded_excel_file):
    try:
        # Define the QA_LLM class
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

                if end_index < start_index or (end_index - start_index) > 50:
                    return "Không tìm thấy thông tin phù hợp trong SOP."

                answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
                return self.tokenizer_qa.decode(answer_tokens, skip_special_tokens=True).strip()

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
            return None, None, None

        xls = pd.ExcelFile(excel_file)
        available_sheets = xls.sheet_names
        required_sheets = ['Tiêu chí giám sát cuộc gọi KH', 'Tiêu chí giám sát cuộc gọi NT']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
        if missing_sheets:
            print(f"Missing sheets: {missing_sheets}")
            return None, None, None

        df_customer_call = pd.read_excel(excel_file, sheet_name='Tiêu chí giám sát cuộc gọi KH')
        df_relative_call = pd.read_excel(excel_file, sheet_name='Tiêu chí giám sát cuộc gọi NT')

        if df_customer_call.empty or df_relative_call.empty:
            print("One or both sheets are empty.")
            return None, None, None

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

        combined_text = "\n".join([doc.page_content for doc in all_docs])
        sop_data = {
            'Tiêu chí giám sát cuộc gọi KH': df_customer_call.astype(str).agg(' '.join, axis=1).tolist(),
            'Tiêu chí giám sát cuộc gọi NT': df_relative_call.astype(str).agg(' '.join, axis=1).tolist()
        }

        # Load the question-answering model and tokenizer
        model_name_qa = "nguyenvulebinh/vi-mrc-large"
        tokenizer_qa = AutoTokenizer.from_pretrained(model_name_qa)
        model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name_qa)

        # Move the model to the appropriate device (GPU or CPU)
        model_qa.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Instantiate the QA_LLM with the model and tokenizer
        qa_llm = QA_LLM(model_qa, tokenizer_qa, combined_text)

        # Create retriever for the vector store DB
        retriever = db.as_retriever()

        return qa_llm, combined_text, sop_data

    except Exception as e:
        print(f"Error loading Excel data: {e}")
        return None, None, None


tokenizer, model, device = load_model()


def safe_tokenize(text, tokenizer, max_length=512):
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
    - If chunk_size is None: classify the whole text.
    - If chunk_size is an integer: split sentences into chunks and classify each chunk with majority voting.
    """
    text = clean_text(text)

    def predict(inputs):
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_id = inputs['input_ids'].max().item()
        if max_id >= model.config.vocab_size:
            return None  

        with torch.no_grad():
            outputs = model(**inputs)
        label = outputs.logits.argmax(dim=1).cpu().item()
        return label

    if chunk_size is None:
        # --- Phân loại nguyên đoạn text ---
        inputs = safe_tokenize(text, tokenizer)
        if inputs is None:
            return [{"text": text, "tone": "Error in tokenization"}]

        label = predict(inputs)
        if label is None:
            return [{"text": text, "tone": "Error: input_ids out of range"}]

        tone = "Hợp tác" if label == 1 else "Không hợp tác"
        return [{"text": text, "tone": tone}]

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

        # Majority voting
        final_label = max(set(labels), key=labels.count)
        tone = "Hợp tác" if final_label == 1 else "Không hợp tác"
        return [{"text": text, "tone": tone}]


def classify_tone_with_llm(text):
    """
    Use GPT (or other LLM) to classify:
    - Is this the customer's sentence?
    - If yes, what is the cooperation tone?
    """

    prompt = f"""
    Bạn là một trợ lý AI cho phân tích cuộc gọi trong thu hồi nợ.

    Câu sau là một phần transcript cuộc gọi:

    "{text}"

    Trả lời bằng JSON với định dạng:
    {{
      "is_customer": true hoặc false,
      "tone": "Hợp tác" | "Không hợp tác" | "Trung lập" | "Không áp dụng"
    }}

    Chỉ trả về JSON, không thêm giải thích.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in tone classification: {e}")
        return {"is_customer": False, "tone": "Không áp dụng", "error": str(e)}


# LLM-based response generator
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


st.title("Đánh giá Cuộc Gọi - AI Bot")

def process_files(uploaded_excel_file, uploaded_audio_file):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Parallel execution of the QA chain and transcription
        future_chain = executor.submit(load_excel_rag_data, uploaded_excel_file)
        future_transcript = executor.submit(transcribe_audio, uploaded_audio_file)

        qa_llm, combined_text, sop_data = future_chain.result()
        transcript = future_transcript.result()

    return qa_llm, combined_text, sop_data, transcript

def main():
    uploaded_excel_file = st.file_uploader("Tải lên tệp Excel", type="xlsx")
    uploaded_audio_file = st.file_uploader("Tải lên tệp âm thanh", type=["mp3", "wav"])


    if uploaded_excel_file and uploaded_audio_file:
        st.success("Tải tệp thành công!")

        if st.button("Đánh giá"):
            with st.spinner("Đang xử lý..."):
                try:
                    qa_chain, combined_text, sop_data, transcript = process_files(uploaded_excel_file, uploaded_audio_file)
                except Exception as e:
                    st.error(f"Lỗi khi xử lý tệp: {e}")
                    return

                st.subheader("Văn bản thu được:")
                st.write(transcript)

          
                analysis_result = analyze_call_transcript(transcript)
                tone_chunks = analysis_result["tone_chunks"]
                customer_label = classify_tone(transcript, chunk_size=3)

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
                    sop_results, sop_rate, sentence_rate, sop_violations = evaluate_sop_compliance(
                        transcript, sop_data, model, threshold=0.8
                    )

                    st.write(f"Tỷ lệ tuân thủ SOP: **{sop_rate:.2f}%**")
                    st.write(f"Tỷ lệ tuân thủ câu nói: **{sentence_rate:.2f}%**")

                    if sop_violations:
                        st.markdown("Các mục chưa tuân thủ:")
                        for idx, sop_item in sop_violations:
                            st.markdown(f"{idx}. {sop_item}")
                    else:
                        st.success("Tất cả các mục trong SOP đã được tuân thủ!")
                except Exception as e:
                    st.error(f"Lỗi khi đánh giá SOP: {e}")

                st.subheader("Phản hồi gợi ý:")
                try:
                    suggestion = suggest_response(transcript, customer_label, use_llm=True)
                    st.write(suggestion)
                except Exception as e:
                    st.error(f"Lỗi khi gợi ý phản hồi: {e}")

                cleanup_memory()

if __name__ == "__main__":
    main()
