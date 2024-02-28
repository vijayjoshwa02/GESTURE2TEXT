import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('CNNmodel.h5')

def prediction(pred):
    return chr(pred + 65)

def keras_predict(model, image):
    data = np.asarray(image, dtype="int32")
    pred_probab = model.predict(data)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

def preprocess_image(img):
    # Resize and preprocess the image for the model
    image_size = (28, 28)
    resized_img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    resized_img = np.resize(resized_img, (*image_size, 1))
    resized_img = np.expand_dims(resized_img, axis=0)
    return resized_img

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    while True:
        # Open the webcam
        cam_capture = cv2.VideoCapture(0)
        _, image_frame = cam_capture.read()  

        # Select ROI
        roi = crop_image(image_frame, 150, 150, 300, 300)

        # Preprocess the ROI for prediction
        preprocessed_image = preprocess_image(roi)

        # Make prediction
        pred_probab, pred_class = keras_predict(model, preprocessed_image)
        curr = prediction(pred_class)

        # Display only the processed ROI with prediction text
        cv2.putText(roi, curr, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow("Processed ROI", roi)

        # Check for user input to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # Release the webcam and close all windows
    cam_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
