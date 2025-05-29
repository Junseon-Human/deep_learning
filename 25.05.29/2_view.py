import cv2
from ultralytics import YOLO  # YOLOv8 라이브러리

# 1. YOLO 모델 불러오기 (사전학습된 모델 사용)
model = YOLO("yolov8n.pt")  # 경량화 모델로 빠르게 확인 가능

# 2. 웹캠 열기 (기본 카메라 index=0)
cap = cv2.VideoCapture(0)

# 3. 반복적으로 프레임 읽고 객체 감지
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO 예측
    results = model(frame, show=False, stream=True)  # stream=True: 실시간 감지

    # 5. 결과 프레임 출력
    for result in results:
        annotated_frame = result.plot()  # 예측결과로 이미지 그리기
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # 6. 종료 조건 (q 키를 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 7. 자원 해제
cap.release()
cv2.destroyAllWindows()
