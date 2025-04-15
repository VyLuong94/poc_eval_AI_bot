import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.base import LLM
import torch
from typing import List

# 1. Load tone classification model
model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def classify_tone(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    label = outputs.logits.argmax(dim=1).cpu().item()
    return "Hợp tác" if label == 1 else "Không hợp tác"

# 2. Load QA chain
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

# Business logic
def suggest_response(text, region, label):
    if label == "Không hợp tác":
        if any(x in text.lower() for x in ["khỏi gọi", "không có tiền"]):
            if region == "Northern":
                return "Ngừng liên hệ trong ngày. Chuyển case giám sát hoặc cảnh báo pháp lý."
            elif region == "Central":
                return "Hỏi nhẹ nhàng thêm lần nữa, nếu không phản hồi, tạm ngưng liên hệ 3 ngày."
            else:
                return "Giữ thái độ nhẹ nhàng, gửi SMS xác nhận và dừng liên lạc trong hôm nay."
        else:
            return "Phản hồi trung lập, cần theo dõi thêm lần liên hệ sau."
    else:
        if region == "Northern":
            return "Cảm ơn khách, xác nhận lại thời gian thanh toán rõ ràng."
        elif region == "Central":
            return "Xác nhận lại cuối tuần, lưu lịch hẹn hệ thống + SMS nhẹ nhàng."
        else:
            return "Cảm ơn khách, gửi SMS xác nhận lịch thanh toán."

def evaluate_agent_text(agent_text):
    issues = []
    if any(x in agent_text.lower() for x in ["phải trả ngay", "nếu không sẽ bị phạt", "tôi sẽ tiếp tục gọi"]):
        issues.append("Nhân viên có thái độ gây áp lực, không phù hợp với SOP.")
    if "không thể chờ đợi" in agent_text.lower():
        issues.append("Nhân viên sử dụng ngôn từ không phù hợp, gây cảm giác không thoải mái cho khách hàng.")
    return "\n".join(issues) if issues else "Nhân viên đã tuân thủ SOP và giữ thái độ chuyên nghiệp."

# Giao diện Streamlit
st.title("POC: Đánh giá hội thoại thu hồi nợ")

customer_text = st.text_area("Nội dung khách hàng", "")
agent_text = st.text_area("Nội dung nhân viên", "")
region = st.selectbox("Vùng miền", ["Northern", "Central", "Southern"])

if st.button("Đánh giá"):
    if customer_text and agent_text:
        label = classify_tone(customer_text)
        agent_eval = evaluate_agent_text(agent_text)
        suggestion = suggest_response(customer_text, region, label)
        sop_answer = qa_chain.run(customer_text)

        st.subheader("Kết quả phân tích:")
        st.write(f"**Phân loại khách hàng:** {label}")
        st.write(f"**Đánh giá nhân viên:** {agent_eval}")
        st.write(f"**Gợi ý phản hồi:** {suggestion}")
        st.write(f"**SOP liên quan:** {sop_answer}")
    else:
        st.warning("Vui lòng nhập đầy đủ nội dung khách hàng và nhân viên.")
