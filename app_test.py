import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.language_models.llms import LLM
from transformers import pipeline
import torch
import time
from typing import List


def load_model():
    model_name = "vyluong/tone-classification-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

def classify_tone(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    label = outputs.logits.argmax(dim=1).cpu().item()
    return "Hợp tác" if label == 1 else "Không hợp tác"

# Cải tiến load RAG QA chain
@st.cache_resource
def load_rag_qa_chain():
    sop_text = """
    1. Nếu khách hàng phản ứng tiêu cực như "không có tiền", "khỏi gọi nữa":
       - Ngừng liên hệ trong ngày.
       - Gửi cảnh báo hoặc chuyển hồ sơ sang bộ phận giám sát.

    2. Nếu khách hàng hợp tác, cam kết thanh toán:
       - Xác nhận lại thời gian thanh toán.
       - Gửi SMS xác nhận cam kết.

    3. Nhân viên cần giữ thái độ bình tĩnh, không gây áp lực khi khách hàng khó chịu.
    4. Lịch liên hệ lại tối đa 3 lần/tuần, không gọi liên tiếp trong 1 ngày.
    """
    docs = [Document(page_content=sop_text)]
    texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
    db = FAISS.from_documents(texts, embedding_model)

    tokenizer_qa = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model_qa = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    class QA_LLM(LLM):
        def _call(self, prompt: str, **kwargs) -> str:
            inputs = tokenizer_qa(prompt, sop_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model_qa(**inputs)
            start_index = torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits)
            if end_index < start_index:
                return "Không tìm thấy thông tin phù hợp trong SOP."
            answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]
            return tokenizer_qa.decode(answer_tokens, skip_special_tokens=True)

        @property
        def _llm_type(self) -> str:
            return "custom-bert-qa"

    custom_llm = QA_LLM()
    return RetrievalQA.from_chain_type(llm=custom_llm, retriever=db.as_retriever())

qa_chain = load_rag_qa_chain()

# Cải tiến LLM pipe
llm_pipe = pipeline("text-generation", model="bigscience/bloomz-560m")

def generate_response(text, region, label):
    prompt = f"""
Bạn là nhân viên chăm sóc khách hàng trong bộ phận thu hồi nợ.

Thông tin khách hàng:
- Khu vực: {region}
- Cảm xúc hiện tại: {label}
- Phát ngôn của khách hàng: "{text}"

Yêu cầu: Trả lời lại khách hàng bằng TIẾNG VIỆT, ngắn gọn (1-2 câu), lịch sự, thân thiện và phù hợp hoàn cảnh để khuyến khích thanh toán. Không được bịa chuyện hay nói điều không liên quan.
Hãy gợi ý một phản hồi ngắn gọn, lịch sự, mang tính thuyết phục hoặc phù hợp hoàn cảnh.
Không nói quá máy móc. Nếu khách hợp tác, nhấn mạnh lịch hẹn. Nếu không hợp tác, chọn hướng xử lý phù hợp.

Trả lời bằng văn phong tự nhiên như người thật:
Phản hồi:
"""
    output = llm_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
    generated = output[0]["generated_text"]
    return generated.replace(prompt, "").strip()

def evaluate_agent_text(agent_text):
    issues = []
    lowered_text = agent_text.lower()

    # Kiểm tra ngôn từ gây áp lực
    if any(x in lowered_text for x in ["phải", "nếu không sẽ bị phạt", "tôi sẽ tiếp tục gọi"]):
        issues.append("Nhân viên có thái độ gây áp lực, không phù hợp với SOP.")

    # Kiểm tra ngôn từ thiếu chuyên nghiệp
    if "không thể chờ đợi" in lowered_text:
        issues.append("Nhân viên sử dụng ngôn từ không phù hợp, gây cảm giác không thoải mái cho khách hàng.")

    # Kiểm tra từ ngữ chửi thề
    bad_words = ["má", "đm", "vcl", "vãi", "mẹ", "vl", "địt", "con chó", "thằng ngu", "con khùng"]  # mở rộng nếu cần
    if any(bad_word in lowered_text for bad_word in bad_words):
        issues.append("Nhân viên sử dụng từ ngữ không phù hợp (chửi thề), vi phạm quy định SOP.")

    return "\n".join(issues) if issues else "Nhân viên đã tuân thủ SOP và giữ thái độ chuyên nghiệp."
# Cải tiến phản hồi thủ công
def rule_based_response(text, region, label):
    text_lower = text.lower()

    if label == "Không hợp tác":
        if any(x in text_lower for x in ["khỏi gọi", "không có tiền", "đừng gọi", "phá sản"]):
            if region == "Northern":
                return "Tạm dừng liên hệ hôm nay, báo cáo cho nhóm giám sát nếu cần."
            elif region == "Central":
                return "Gửi lời cảm ơn, nhắc nhẹ và tạm dừng liên hệ trong 3 ngày."
            else:  # Southern
                return "Gửi SMS xác nhận đã ghi nhận ý kiến khách và tạm ngưng liên hệ hôm nay."
        else:
            return "Chưa có tín hiệu rõ ràng, cần theo dõi thêm ở lần liên hệ tiếp theo."
    else:
        if region == "Northern":
            return "Cảm ơn anh/chị, xác nhận thời gian thanh toán rõ ràng để hỗ trợ tốt nhất."
        elif region == "Central":
            return "Ghi nhận ý kiến, xác nhận lịch hẹn cuối tuần và gửi SMS nhắc nhẹ."
        else:  # Southern
            return "Cảm ơn anh/chị đã phối hợp, hệ thống sẽ gửi lại xác nhận lịch thanh toán."

# Kiểm tra thời gian của mỗi tác vụ
@st.cache_data
def eval_conversation(customer_text, agent_text, region, use_llm=True):
    try:
        start_time = time.time()  # Ghi lại thời gian bắt đầu
        
        # Các xử lý khác của ứng dụng...
        label = classify_tone(customer_text)
        agent_eval = evaluate_agent_text(agent_text)
        suggestion = suggest_response(customer_text, region, label, use_llm=use_llm)
        sop_answer = qa_chain(customer_text)

        elapsed_time = time.time() - start_time  # Tính thời gian xử lý
        st.write(f"Thời gian xử lý: {elapsed_time:.2f} giây.")  # Hiển thị thời gian

        return label, agent_eval, suggestion, sop_answer
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")  # Hiển thị thông báo lỗi nếu có
        raise e

def suggest_response(text, region, label, use_llm=True):
    if use_llm:
        return generate_response(text, region, label)
    else:
        return rule_based_response(text, region, label)


# UI Streamlit
st.title("POC: Đánh giá hội thoại thu hồi nợ")

customer_text = st.text_area("Nội dung khách hàng", "")
agent_text = st.text_area("Nội dung nhân viên", "")
region = st.selectbox("Vùng miền", ["Northern", "Central", "Southern"])

if st.button("Đánh giá"):
    if customer_text and agent_text:
        label, agent_eval, suggestion, sop_answer = eval_conversation(customer_text, agent_text, region)

        st.subheader("Kết quả phân tích:")
        st.write(f"**Phân loại khách hàng:** {label}")
        st.write(f"**Đánh giá nhân viên:** {agent_eval}")
        st.write(f"**Gợi ý phản hồi:** {suggestion}")
        st.write(f"**SOP liên quan:** {sop_answer}")
    else:
        st.warning("Vui lòng nhập đầy đủ nội dung khách hàng và nhân viên.")
