# main.py

# 1. 필요한 라이브러리 임포트
from sentence_transformers import SentenceTransformer
import sqlite3
from datetime import datetime
import re # 정규표현식 모듈 (사용자 입력 파싱에 유용)
from torch.nn.functional import cosine_similarity # 유사도 계산 함수를 명시적으로 임포트

# 2. 봇의 지식 베이스 정의 (초기 버전은 간단한 딕셔너리)
knowledge_base = {
    "피타고라스 정리": "직각 삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다. 공식은 a² + b² = c² 입니다.",
    "파이썬 변수": "파이썬에서 변수는 데이터를 저장하는 공간을 의미합니다. 예를 들어, 'x = 10'에서 'x'는 변수입니다.",
    "머신러닝": "머신러닝은 컴퓨터가 데이터를 학습하여 패턴을 인식하고 예측을 수행하는 인공지능의 한 분야입니다.",
    "인공지능": "인공지능(AI)은 인간의 학습 능력, 추론 능력, 지각 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
    "데이터 과학": "데이터 과학은 데이터로부터 지식과 통찰력을 추출하는 학문 분야입니다. 통계학, 컴퓨터 과학, 도메인 지식이 결합됩니다."
}

# SentenceTransformer 모델 로드 (한국어 모델)
try:
    # 모델 이름을 'jhgan/ko-sroberta-multitask'로 변경
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    print("AI 모델 로드 완료.")
except Exception as e:
    print(f"AI 모델 로드 중 오류 발생: {e}")
    print("모델 없이 기본적인 키워드 매칭으로 작동합니다. (성능 제한적)")
    model = None

# 지식 베이스의 모든 문장을 임베딩 (모델이 로드된 경우에만)
knowledge_texts = list(knowledge_base.keys()) + list(knowledge_base.values())
knowledge_embeddings = None
if model:
    print("지식 베이스 임베딩 중...")
    knowledge_embeddings = model.encode(knowledge_texts, convert_to_tensor=True)
    print("지식 베이스 임베딩 완료.")

# 명령어 패턴 정의 및 임베딩 (AI 모델이 로드된 경우에만)
# 각 명령어 기능에 대해 다양한 표현의 예시 문장들을 정의합니다.
command_patterns = {
    "save_wrong_answer": [
        "오답 노트에 저장해줘:", "오답 정리해줘:", "틀린 문제 기록:",
        "이거 오답이야:", "헷갈리는 개념 저장해줘:", "오답으로 남길래:"
    ],
    "delete_wrong_answer": [
        "오답 노트에서 삭제해줘:", "오답 지워줘:", "틀린 문제 삭제:",
        "이 오답 지워줘:", "오답 기록 없애줘:", "오답 노트에서 지워줘:"
    ],
    "get_wrong_answers": [
        "내 오답 노트 보여줘", "오답 노트 확인", "틀린 문제 뭐 있어?",
        "오답 목록 보여줘", "오답 노트 읽어줘", "내가 틀린 거 알려줘"
    ],
    "save_learning_progress": [
        "학습 진도 기록해줘:", "오늘 공부한 거 기록:", "학습 기록 남겨줘:",
        "내 진도 저장해줘:", "공부한 거 기록해줘:", "오늘 배운 거 저장:"
    ],
    "get_learning_progress": [
        "내 학습 진도 보여줘", "오늘 뭐 공부했어?", "학습 진도 확인",
        "내 공부 기록 보여줘", "나 얼마나 공부했어?", "진도 알려줘"
    ],
    "delete_learning_progress": [
        "학습 진도 삭제해줘:", "공부 기록 지워줘:", "진도 기록 삭제:",
        "이 진도 없애줘:", "학습 진도 지워줘:", "진도 내역 삭제:"
    ],
    "exit": [
        "종료", "봇 종료", "그만", "나가기", "이제 그만", "대화 끝"
    ]
}

command_embeddings = {}
if model: # 모델이 정상적으로 로드되었을 때만 임베딩
    print("명령어 패턴 임베딩 중...")
    for cmd_type, patterns in command_patterns.items():
        command_embeddings[cmd_type] = model.encode(patterns, convert_to_tensor=True)
    print("명령어 패턴 임베딩 완료.")


# 3. 데이터베이스 초기화 및 테이블 생성 함수
def init_db():
    conn = sqlite3.connect('learning_assistant.db') # 데이터베이스 파일 생성 또는 연결
    cursor = conn.cursor()

    # 오답 노트 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wrong_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            content TEXT NOT NULL
        )
    ''')

    # 학습 진도 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            topic TEXT NOT NULL
        )
    ''')

    conn.commit() # 변경사항 저장
    conn.close() # 연결 종료
    print("데이터베이스 초기화 완료.")

# 4. 주요 기능 함수 정의

# 4.1. 질의응답 함수
def get_answer(question):
    if not model or knowledge_embeddings is None:
        # 모델 로드에 실패했거나 임베딩이 없는 경우, 간단한 키워드 매칭 시도
        for key, value in knowledge_base.items():
            if question.lower() in key.lower() or question.lower() in value.lower():
                return value
        return "죄송합니다. 아직 해당 질문에 대한 정보가 부족합니다. (모델 로드 실패 또는 키워드 불일치)"

    # 질문 임베딩
    question_embedding = model.encode(question, convert_to_tensor=True)

    # 유사도 계산 (코사인 유사도)
    # question_embedding과 knowledge_embeddings의 차원 구조를 확인하고,
    # 유사도 계산 함수에 바로 넘겨줍니다.
    # 일반적으로 SentenceTransformer의 encode 결과는 (embedding_dim,) 이므로
    # 단일 질문인 경우 unsqueeze(0)를 한 번만 사용해서 (1, embedding_dim) 형태로 맞춰주는 것이 안전합니다.
    similarities = cosine_similarity(question_embedding.unsqueeze(0), knowledge_embeddings)
    # 결과가 (1, num_knowledge_texts) 형태일 것이므로, 단일 벡터로 만들기 위해 squeeze()
    similarities = similarities.squeeze(0) # 첫 번째 차원(1)을 제거합니다.

    # 가장 유사한 문장 찾기
    max_similarity_index = similarities.argmax().item()
    most_similar_text_in_corpus = knowledge_texts[max_similarity_index]
    max_similarity_score = similarities[max_similarity_index].item()

    # 유사도 점수가 낮으면 관련 없는 질문으로 판단
    if max_similarity_score < 0.6: # 이 임계값은 조정 가능합니다.
        return "죄송합니다. 질문하신 내용과 관련된 정보를 찾기 어렵습니다. 좀 더 구체적으로 질문해주시겠어요?"

    # 가장 유사한 텍스트가 knowledge_base의 키에 있는지 확인하여 답변 반환
    if most_similar_text_in_corpus in knowledge_base:
        return knowledge_base[most_similar_text_in_corpus]
    else:
        # 유사한 텍스트가 값(설명) 부분에 있다면, 해당 값의 키를 찾아 답변 반환
        for key, value in knowledge_base.items():
            if most_similar_text_in_corpus == value:
                return value # 또는 key에 해당하는 답변을 반환하도록 로직 개선 (예: "{}에 대한 설명입니다: {}".format(key, value))
        return "죄송합니다. 해당 질문에 대한 답변을 찾았지만, 정확한 내용을 제공하기 어렵습니다."


# 4.2. 오답 노트 저장 함수
def save_wrong_answer(content):
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("INSERT INTO wrong_answers (date, content) VALUES (?, ?)", (today, content))
        conn.commit()
        return "오답 노트에 저장했습니다."
    except sqlite3.Error as e:
        conn.rollback()
        return f"오답 노트 저장 중 오류 발생: {e}"
    finally:
        conn.close()

# 4.2.1 오답 노트 삭제 함수
def del_wrong_answer(content):
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM wrong_answers WHERE content=?", (content,))
        conn.commit()
        deleted_rows = cursor.rowcount
        if deleted_rows > 0:
            return f"'{content}' 내용을 포함하는 오답 노트 {deleted_rows}개를 삭제했습니다."
        else:
            return f"'{content}' 내용을 포함하는 오답 노트를 찾을 수 없습니다."
    except sqlite3.Error as e:
        conn.rollback()
        return f"오답 노트 삭제 중 오류 발생: {e}"
    finally:
        conn.close()

# 4.2.2 오답 노트 확인 함수
def get_wrong_answers():
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, content FROM wrong_answers ORDER BY date DESC")
    wrong_answers_records = cursor.fetchall()
    conn.close()

    if wrong_answers_records:
        response = "현재 오답 노트:\n"
        for id, date, content in wrong_answers_records:
            response += f"- ID: {id}, 날짜: {date}, 내용: {content}\n"
        return response
    else:
        return "아직 기록된 오답 노트가 없습니다."


# 4.3. 학습 진도 기록 함수
def save_learning_progress(topic):
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("INSERT INTO learning_progress (date, topic) VALUES (?, ?)", (today, topic))
        conn.commit()
        return "학습 진도를 기록했습니다."
    except sqlite3.Error as e:
        conn.rollback()
        return f"학습 진도 저장 중 오류 발생: {e}"
    finally:
        conn.close()

# 4.4. 학습 진도 확인 함수
def get_learning_progress():
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    cursor.execute("SELECT date, topic FROM learning_progress ORDER BY date DESC LIMIT 5") # 최근 5개만 조회
    progress_records = cursor.fetchall()
    conn.close()

    if progress_records:
        response = "최근 학습 진도:\n"
        for date, topic in progress_records:
            response += f"- {date}: {topic}\n"
        return response
    else:
        return "아직 기록된 학습 진도가 없습니다."

# 4.5 학습 진도 삭제 함수
def del_learning_progress(topic):
    conn = sqlite3.connect('learning_assistant.db')
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM learning_progress WHERE topic=?", (topic,))
        conn.commit()
        deleted_rows = cursor.rowcount
        if deleted_rows > 0:
            return f"'{topic}' 내용을 포함하는 학습 진도 {deleted_rows}개를 삭제했습니다."
        else:
            return f"'{topic}' 내용을 포함하는 학습 진도를 찾을 수 없습니다."
    except sqlite3.Error as e:
        conn.rollback()
        return f"학습 진도 삭제 중 오류 발생: {e}"
    finally:
        conn.close()


# 5. 봇의 메인 루프 (사용자 입력 처리)
def run_bot():
    init_db() # 봇 시작 시 데이터베이스 초기화/생성

    print("--- 개인 맞춤형 학습 도우미 봇입니다 ---")
    print("명령어 예시: 이제 더 자연스럽게 말해보세요!")
    print("  - 질문: '피타고라스 정리가 뭐야?'")
    print("  - 오답 저장: '오답 노트에 저장해줘: 로그 개념이 어려워'")
    print("  - 오답 삭제: '오답 노트에서 삭제해줘: 로그 개념이 어려워'")
    print("  - 오답 확인: '내 오답 노트 보여줘'")
    print("  - 진도 기록: '학습 진도 기록해줘: 파이썬 변수'")
    print("  - 진도 확인: '내 학습 진도 보여줘'")
    print("  - 진도 삭제: '학습 진도 삭제해줘: 파이썬 변수'")
    print("  - 종료: '종료'")
    print("------------------------------------")

    while True:
        user_input = input(">> 당신의 질문/명령: ").strip()

        # 종료 명령은 AI 분석을 거치지 않고 먼저 처리
        if user_input.lower() in command_patterns["exit"]:
            print("봇을 종료합니다. 다음에 또 만나요!")
            break

        # AI 모델이 로드되지 않았다면 제한적인 기능만 사용
        if not model or not command_embeddings:
            print("AI 모델이 로드되지 않아 제한적인 기능만 사용 가능합니다. 질문이나 '종료'만 가능합니다.")
            answer = get_answer(user_input) # 기존 get_answer는 키워드 매칭 fallback이 있음
            print(answer)
            continue

        # 사용자 입력 임베딩
        user_input_embedding = model.encode(user_input, convert_to_tensor=True).unsqueeze(0)
        
        best_command_type = None
        max_command_similarity = -1.0 # 유사도 점수는 -1에서 1 사이

        # 각 명령어 타입별로 유사도 계산
        for cmd_type, patterns_embeddings in command_embeddings.items():
            # 사용자의 입력과 해당 명령어 타입의 모든 패턴 임베딩 간 유사도 계산
            similarities = cosine_similarity(user_input_embedding, patterns_embeddings) # (1, N)
            
            # 해당 명령어 타입 내에서 가장 높은 유사도 점수 찾기
            current_max_similarity = similarities.max().item()

            if current_max_similarity > max_command_similarity:
                max_command_similarity = current_max_similarity
                best_command_type = cmd_type
        
        # 임계값 설정 (이 값을 조정하여 명령어 인식의 민감도를 조절합니다)
        command_threshold = 0.5 # 0.4 ~ 0.7 사이에서 테스트하며 적절한 값 찾기

        if best_command_type and max_command_similarity >= command_threshold:
            # 명령어 내용 추출 로직 개선: 콜론(:)을 기준으로 내용을 분리
            # 콜론이 없는 명령어 (예: '내 오답 노트 보여줘')는 추출된 내용이 없습니다.
            extracted_content = ""
            if ':' in user_input:
                # 첫 번째 콜론 이후의 모든 텍스트를 내용으로 간주
                extracted_content = user_input.split(':', 1)[1].strip()

            # 특정 명령어에 매칭될 경우 해당 기능 실행
            if best_command_type == "get_wrong_answers":
                print(get_wrong_answers())
            elif best_command_type == "get_learning_progress":
                print(get_learning_progress())
            elif best_command_type == "save_wrong_answer":
                if extracted_content:
                    print(save_wrong_answer(extracted_content))
                else:
                    print("오답 노트에 저장할 내용을 입력해주세요. 예: '오답 노트에 저장해줘: 로그 개념이 어려워'")
            elif best_command_type == "delete_wrong_answer":
                if extracted_content:
                    print(del_wrong_answer(extracted_content))
                else:
                    print("삭제할 오답 노트 내용을 입력해주세요. 예: '오답 노트에서 삭제해줘: 로그 개념이 어려워'")
            elif best_command_type == "save_learning_progress":
                if extracted_content:
                    print(save_learning_progress(extracted_content))
                else:
                    print("학습 진도 내용을 입력해주세요. 예: '학습 진도 기록해줘: 파이썬 변수'")
            elif best_command_type == "delete_learning_progress":
                if extracted_content:
                    print(del_learning_progress(extracted_content))
                else:
                    print("삭제할 학습 진도 내용을 입력해주세요. 예: '학습 진도 삭제해줘: 파이썬 변수'")
            # 'exit' 명령어는 이미 위에서 처리했으므로 여기서는 추가적으로 처리할 필요 없음
            else:
                 print("알 수 없는 명령어 유형입니다. 다시 시도해주세요.") # 예상치 못한 명령어 타입
        else:
            # 명령어와 유사도가 낮거나 매칭되는 명령어가 없을 경우 질문으로 간주
            answer = get_answer(user_input)
            print(answer)

# 스크립트가 직접 실행될 때만 run_bot() 함수 호출
if __name__ == "__main__":
    run_bot()