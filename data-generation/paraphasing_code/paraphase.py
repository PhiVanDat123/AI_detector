from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer
import os
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

embedding_model = SentenceTransformer("BAAI/bge-m3")

system_prompt = "Bạn là một trợ lý cực kỳ thông minh, chu đáo và chính xác. Mục tiêu của bạn là hiểu sâu sắc ý định của người dùng, đặt câu hỏi làm rõ khi cần thiết, suy nghĩ từng bước qua các vấn đề phức tạp, cung cấp câu trả lời rõ ràng và chính xác, đồng thời chủ động dự đoán thông tin bổ sung hữu ích. Luôn ưu tiên sự thật, tính chính xác, cái nhìn sâu sắc và hiệu quả, điều chỉnh phản hồi của bạn sao cho phù hợp nhất với nhu cầu và sở thích cụ thể của người dùng."

def compare_sim(p1, p2, threshold=0.8):
    b1 = embedding_model.encode(p1)
    b2 = embedding_model.encode(p2)
    sim = embedding_model.similarity(b1, b2)
    return sim[0][0]


def paraphase_function(title, paragraph, pid, model, base_url, retry_num=5):

    client = OpenAI(
        api_key=os.environ.get(model),
        base_url=base_url
    )

    prompt = f"""
    #### **Hướng dẫn**  
    Diễn đạt lại nội dung luận án được cung cấp trong khi vẫn giữ nguyên ý nghĩa và bối cảnh ban đầu. Đảm bảo sự rõ ràng, mạch lạc và duy trì giọng văn học thuật.  

    XUẤT RA CHỈ NỘI DUNG ĐÃ DIỄN ĐẠT LẠI, KHÔNG CÓ PHẦN GIỚI THIỆU HOẶC GIẢI THÍCH.  

    KẾT QUẢ CHỈ ĐƯỢC VIẾT DƯỚI DẠNG MỘT ĐOẠN VĂN DUY NHẤT.  

    KHÔNG DIỄN ĐẠT LẠI: Tài liệu tham khảo, nhãn hình (ví dụ: Hình A, B, C) và trích dẫn phải được giữ nguyên.  

    #### **Ngữ cảnh (Tiêu đề luận án)**  
    "{title}"  

    #### **Đoạn văn đầu vào**  
    "{paragraph}"  

    #### **Định dạng đầu ra**  
    [NỘI DUNG ĐÃ DIỄN ĐẠT LẠI]
    """
    for attempt in range(retry_num):
        try:
            result = client.chat.completions.create(
                model=model,
                temperature=0.3+attempt*0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            response = result.choices[0].message.content
            
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed - ID: {id}, model: {model}, Error: {str(e)}")
            
            # Exponential backoff: 2^attempt seconds (2,4,8,16,32 seconds)
            wait_time = 2 * attempt
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            
            # If this was the last attempt, raise the error
            if attempt == retry_num - 1:
                raise Exception(f"All {retry_num} attempts failed for ID: {id}, model: {model}, last error: {str(e)}")

