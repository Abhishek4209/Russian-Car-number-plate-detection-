import cv2

# Path to the Haar Cascade classifier XML file for license plate detection
harcascade = "D:\Car Number Plate detection\model\haarcascade_russian_plate_number.xml"

# Accessing the webcam (index 0) using VideoCapture
cap = cv2.VideoCapture(0)

# Setting the dimensions for the video stream
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum area for a detected region to be considered a license plate
min_area = 500
count = 0

# Creating a license plate classifier
plate_cascade = cv2.CascadeClassifier(harcascade)

while True:
    # Reading frames from the webcam
    success, img = cap.read()

    # Converting the frame to grayscale for easier processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting license plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterating through the detected plates
    for (x, y, w, h) in plates:
        area = w * h

        # Checking if the detected area is larger than the minimum area
        if area > min_area:
            # Drawing a rectangle around the detected license plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Adding a label for the detected plate
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extracting the region of interest (ROI) for the license plate
            img_roi = img[y: y + h, x:x + w]

            # Displaying the ROI in a separate window
            cv2.imshow("ROI", img_roi)

    # Displaying the result with detected plates
    cv2.imshow("Result", img)

    # Saving the plate when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Writing the plate image to a file with a count in the filename
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)

        # Displaying a message that the plate is saved
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)

        # Waiting for 500 milliseconds before continuing
        cv2.waitKey(500)

        # Incrementing the count for the next saved plate
        count += 1

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture
cap.release()
cv2.destroyAllWindows()
