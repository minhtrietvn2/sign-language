# Phần import thư viện
import cv2
import os
import numpy as np
import imutils
# Biến global
background = None
name = 'z'
folder_save = 'E:/project hang/project/data_create/' + name
# -------------------------------------
# Phần chương trình con
# -------------------------------------
def run_avg(image, aWeight):
    # Chương trình xử lý background và subtraction background.
    global background
    if background is None:
            background = image.copy().astype("float")
            return
    cv2.accumulateWeighted(image, background, aWeight)  # Dòng lệnh dùng để subtraction
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
    while(True):
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
        if num_frame < 30:
            run_avg(gray,aWeight)
        else:
            # segment ra vùng bàn tay
            hand = segment(gray)
            # kiểm tra vùng bàn sau segment
            if hand is not None:
                # Nếu có thì lưu vào 2 biến thresholed và segmented
                (thresholded, segmented) = hand
                # vẽ vùng segment và hiển thị lên frame
                cv2.drawContours(Clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Threshoh",thresholded)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # Phím thoát chương trình
            print("Thoat chuong trinh\n")
            break
        elif key == ord("r"):
            # Phím reset background
            print(" Da reset backgournd\n")
            background = None
            num_frame = 0
        elif key == ord("s"):
            # Phím save ảnh
            img_name = name+'_{}.jpg'.format(img_counter)
            cv2.imwrite(os.path.join(folder_save, img_name), thresholded)
            print("{} written!".format(img_name))
            img_counter += 1
        cv2.rectangle(Clone,(left,top),(right,bottom),(255,0,0),5)
        num_frame+=1
        cv2.imshow("Camera",Clone)
    # Giải phóng bộ nhớ
    camera.release()
    cv2.destroyAllWindows()






