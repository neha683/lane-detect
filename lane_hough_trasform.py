import sys
import os
import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0] * 0.75)
    y2 = int(image.shape[0] * 0.65 + 0.5)
    x1 = int((y1 - intercept)/slope + 0.5)
    x2 = int((y2 - intercept)/slope + 0.5)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    left_line = np.array([0, 0, 0, 0])
    right_line = np.array([0, 0, 0, 0])
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if np.abs(slope) < 0.15:
                continue
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) != 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    if len(right_fit) != 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 30)
    return canny

def region_of_interest(f, image):
    height = image.shape[0]
    width = image.shape[1]
    if f == 'sunny.mp4':
        roi = np.array([
            [(int(1.5*width/6 + 0.5), int(4.8 * height/6 + 0.5)), (int(3.8*width/6 + 0.5), int(4.8 * height/6 + 0.5)), (int(3*width/6 + 0.5), int(3.5*height/6 + 0.5))]
        ])
    elif f == 'rainy.mp4':
        roi = np.array([
            [(int(0*width/6 + 0.5), int(4.8 * height/6 + 0.5)), (int(3.9*width/6 + 0.5), int(4.8 * height/6 + 0.5)), (int(3*width/6 + 0.5), int(4.3*height/6 + 0.5))]
        ])
    else:
        roi = np.array([
            [(0, int(4 * height/5 + 0.5)), (int(5*width/6 + 0.5), int(4 * height/5 + 0.5)), (int(width/2 + 0.5), int(height/2.8 + 0.5))]
        ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def main(file_name):
    cap = cv2.VideoCapture(file_name)
    while(True):
        ret, frame = cap.read()

        if ret is False:
            break

        lane_image = np.copy(frame)
        edges = canny(lane_image)
        cropped_image = region_of_interest(file_name, edges)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=30)
        #averaged_lines = average_slope_intercept(lane_image, lines)
        #line_image = display_lines(lane_image, averaged_lines)
        line_image = display_lines(lane_image, lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        cv2.imshow('Frame', combo_image)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = sys.argv[1]
        if os.path.isfile(f) and os.path.exists(f):
            main(f)
        else:
            print('provide video file name as argument')
    else:
        print ('provide video file name as argument')
