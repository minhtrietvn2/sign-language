#  Phần quản lý thư viện:
import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import imutils


# Phần quản lý các biến:
predict_Threshold = 90
isBgcaptured = 0
background = None
prediction = ''             # biến dự đoán
score = 0
name = 'mymodel_2.h5'       # Tên model
gesture_names = {0:'A',  1:'B',  2:'C',  3:'D',  4:'E',  5:'F',
                 6:'G',  7:'H',  8:'I',  9:'K',  10:'L', 11:'M',
                 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S',
                 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'}

# Load model:
model = load_model('models/mymodel_18_12_2020.h5')
#   Vùng ROI:
cap_x_begin = 0.5
cap_y_end = 0.5
#   Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.01)

# -------------------------------------
# Phần chương trình con
# -------------------------------------
def predict_image_vgg(image):
    # Chuyen hinh anh thanh ma tran va chuyen ve kieu du lieu float32
    image = np.array(image,dtype='float32')
    image /= 255
    predict_array = model.predict(image)
    print(f'predict_array: {predict_array}')
    result = gesture_names[np.argmax(predict_array)]
    print(f'Result: {result}')
    print(max(predict_array[0]))
    score = float ("%0.2f"%(max(predict_array[0])*100))
    print(result)
    return result,score

def run_avg(image, aWeight):
    # Chương trình xử lý background
    global background
    if background is None:
            background = image.copy().astype("float")
            return
    cv2.accumulateWeighted(image, background, aWeight)

# Segment bàn tay
def segment(image, threshold=25):
    global background
    # Tìm điểm khác biệt giữa ảnh nền và frame hiện tại
    diff = cv2.absdiff(background.astype("uint8"), image)
    # Lấy ngưỡng các điểm khác biệt.
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # Lấy các đường viền trong ảnh đã lấy ngưỡng.
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Trả về None nếu như không phát hiện bất cứ contour nào.
    if len(cnts) == 0:
        return
    else:
        # Dưa trên giá trị của contour, lấy vùng bàn tay.
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
# = ----------------
#   MAIN FUNCTION
# = ----------------
if __name__ == "__main__":
    # Biến
    img_counter = 0                              # Biến đếm dùng để lưu ảnh
    aWeight = 0.9                                # Khởi tạo biến aWeight
    camera = cv2.VideoCapture(0)                 # Khởi tạo để nhận hình ảnh từ camera (webcam)
    num_frame = 0                                # Khởi tạo biến số frame
    top,right,bottom,left = 0,350,250,640        # Tạo giá trị cho vùng ROI
    while camera.isOpened():
        # Lấy frame hiện tại
        (grabbed,frame) = camera.read()
        # resize frame ảnh
        frame = imutils.resize(frame,width=700)
        # lật khung
        frame = cv2.flip(frame,1)
        #clone frame
        Clone = frame.copy()
        #lấy độ dài rộng của frame
        (height,weight) = frame.shape[:2]
        # tạo vùng ROI
        ROI = frame[top:bottom,right:left]
        # chuyển đổi vùng ROI sang gray và làm mờ nó
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Xử lý phím bấm
        if isBgcaptured ==1:
            if num_frame < 30:
                run_avg(gray,aWeight)
            else:
                # segment ra vùng bàn tay
                hand = segment(gray)
                # kiểm tra vùng bàn sau segment
                if (hand is not None):
                    # Nếu có thì lưu vào 2 biến thresholed và segmented
                    (thresholded, segmented) = hand
                    # vẽ vùng segment và hiển thị lên frame
                    cv2.drawContours(Clone, [segmented + (right, top)], -1, (0, 0, 255))
                    cv2.imshow("Thresholed",thresholded)
                    target = np.stack((thresholded,) * 3, axis=-1)
                    target = cv2.resize(target,(224, 224))  # Resize về kích cỡ 224*224 vì sử dụng mạng VGG16
                    target = target.reshape(1, 224, 224, 3)  # Chuyển về 3 kênh màu.
                    prediction, score = predict_image_vgg(target)  # Trả ra giá trị

                    # Nếu độ chính xác >= ngưỡng dự đoán thì xuất kết quả ra camera.
                    print(score, prediction)
                    if (score >= predict_Threshold):
                            cv2.putText(Clone, "Ky tu:" + prediction,(int(cap_x_begin * frame.shape[1]) - 5, int(cap_y_end * frame.shape[0]) + 50),cv2.FONT_ITALIC, 2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
            thresholded = None
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # Phím thoát chương trình
            print("Thoat chuong trinh\n")
            break
        elif key == ord('b'):
            isBgcaptured = 1
            print(" Da chon background\n")
            num_frame = 0
        elif key == ord("r"):
            # Phím reset background
            print(" Da reset backgournd\n")
            isBgcaptured = 0
            background = None
            num_frame = 0
        cv2.rectangle(Clone,(left,top),(right,bottom),(255,0,0),5)
        num_frame+=1
        cv2.imshow("Camera",Clone)
    # Giải phóng bộ nhớ
camera.release()
cv2.destroyAllWindows()







