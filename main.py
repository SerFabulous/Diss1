import cv2
import numpy as np
from rubik_solver import utils


#       Y
#   B   R   G   O
#       W
#
#   Order of Face Input: Y, B, R, G, O, W
#   Faces must be input from Centre Perspective: R, Rotate around R


# Threshold arrays Order: H S V T - Min Max
YellowSquare = [[20,35],[55,100],[130,255],[160,235]]
BlueSquare = [[95,115],[180,255],[120,200],[160,235]]
RedSquare = [[0,25],[100,230],[165,200],[160,235]]
GreenSquare = [[55,85],[55,130],[110,210],[160,235]]
OrangeSquare = [[5,25],[90,175],[240,255],[160,235]]
WhiteSquare = [[20,30],[10,60],[240,255],[160,235]]

# Array for access to Thresholds
SquareColourValues = [YellowSquare,BlueSquare, RedSquare, GreenSquare, OrangeSquare, WhiteSquare]
SquareColourNames = ["y", "b", "r", "g", "o", "w"]

# Arrays to hold Face Config
YellowFace = ["","","","","","","","",""]
BlueFace = ["","","","","","","","",""]
RedFace = ["","","","","","","","",""]
GreenFace = ["","","","","","","","",""]
OrangeFace = ["","","","","","","","",""]
WhiteFace = ["","","","","","","","",""]

# Array for access to Faces
FaceArray = [YellowFace, BlueFace, RedFace, GreenFace, OrangeFace, WhiteFace]

# A Function must be passed when creating track bars
# No action needs to take place here
def empty(a):
    pass

# Find Contours in passed image
# Ensure Contours are of minimum size to avoid false positives
def getContours(imgCanny):
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:

            # This code is useful for testing

            #print(area)
            #cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)

            #peri = cv2.arcLength(cnt, True)
            #approx = cv2.approxPolyDP(cnt, 0.1*peri, True) #   When approx has four points, is size four. Recognise as square

            #cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            #print(approx)
            #objCor = len(approx)

            #x, y, w, h = cv2.boundingRect(approx)
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            return True




# Set Frame Dimentions 
# Get image feed
FRAMEWIDTH = 640
FRAMEHEIGHT = 480
cap = cv2.VideoCapture(0)
cap.set(3, FRAMEWIDTH)
cap.set(4, FRAMEHEIGHT)
WAITKEYDELAY = 10

# Set Dimensions for ROIS and BoundBoxes
SIDELENGTH = 50
OFFSETVALUE = 65
CENTERWIDTH = int(FRAMEWIDTH / 2)
CENTERHEIGHT = int(FRAMEHEIGHT / 2)
RIGHTOFFSET = CENTERWIDTH + OFFSETVALUE
LEFTOFFSET = CENTERWIDTH - OFFSETVALUE
TOPOFFSET = CENTERHEIGHT - OFFSETVALUE
BOTTOMOFFSET = CENTERHEIGHT + OFFSETVALUE

# Create Indices for State and Colour
# Initialise to zero
colourIndex = 0
faceIndex = 0
cubeString = ""
cubeSolution = ""
displayMessage = ""

# Create TrackBars
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 300)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

cv2.createTrackbar("Threshold1", "TrackBars", 160, 255, empty)
cv2.createTrackbar("Threshold2", "TrackBars", 235, 255, empty)

# Loop Through Colour Presets, Find match
while True:
    success, img = cap.read()

    # Get Track Bars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    threshold1 = cv2.getTrackbarPos("Threshold1", "TrackBars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "TrackBars")

    # Create Mask Boundaries
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    #   Draw Nine boxes - Placing them before processing bakes into image
    #   Used to create distinct lines to detect

    # Top
    cv2.rectangle(img, (LEFTOFFSET, TOPOFFSET), (LEFTOFFSET + SIDELENGTH, TOPOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (CENTERWIDTH, TOPOFFSET), (CENTERWIDTH + SIDELENGTH, TOPOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (RIGHTOFFSET, TOPOFFSET), (RIGHTOFFSET + SIDELENGTH, TOPOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)
    # Mid
    cv2.rectangle(img, (LEFTOFFSET, CENTERHEIGHT), (LEFTOFFSET + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (CENTERWIDTH, CENTERHEIGHT), (CENTERWIDTH + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (RIGHTOFFSET, CENTERHEIGHT), (RIGHTOFFSET + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                  (0, 0, 0), 2)
    # Bot
    cv2.rectangle(img, (LEFTOFFSET, BOTTOMOFFSET), (LEFTOFFSET + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (CENTERWIDTH, BOTTOMOFFSET), (CENTERWIDTH + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (RIGHTOFFSET, BOTTOMOFFSET), (RIGHTOFFSET + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                  (0, 0, 0), 2)



    # HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    # Canny for a balance between edge preservation and speed, Bilateral edge preservation at the cost of speed
    # Blue removes noise
    imgBlur = cv2.GaussianBlur(mask, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)

    # For Testing
    # Uncomment this section to manually test preset HSV
    #cv2.setTrackbarPos("Hue Min", "TrackBars", int(SquareColourValues[2][0][0]))
    #cv2.setTrackbarPos("Hue Max", "TrackBars", int(SquareColourValues[2][0][1]))
    #cv2.setTrackbarPos("Sat Min", "TrackBars", int(SquareColourValues[2][1][0]))
    #cv2.setTrackbarPos("Sat Max", "TrackBars", int(SquareColourValues[2][1][1]))
    #cv2.setTrackbarPos("Val Min", "TrackBars", int(SquareColourValues[2][2][0]))
    #cv2.setTrackbarPos("Val Max", "TrackBars", int(SquareColourValues[2][2][1]))

    # Set HSV Thresholds By values set from colourIndex
    cv2.setTrackbarPos("Hue Min", "TrackBars", int(SquareColourValues[colourIndex][0][0]))
    cv2.setTrackbarPos("Hue Max", "TrackBars", int(SquareColourValues[colourIndex][0][1]))
    cv2.setTrackbarPos("Sat Min", "TrackBars", int(SquareColourValues[colourIndex][1][0]))
    cv2.setTrackbarPos("Sat Max", "TrackBars", int(SquareColourValues[colourIndex][1][1]))
    cv2.setTrackbarPos("Val Min", "TrackBars", int(SquareColourValues[colourIndex][2][0]))
    cv2.setTrackbarPos("Val Max", "TrackBars", int(SquareColourValues[colourIndex][2][1]))

    # For Each ROI, look for Contours, if found - record in array, signal to user
    #   Top
    if getContours(imgCanny[TOPOFFSET: TOPOFFSET + SIDELENGTH, LEFTOFFSET: LEFTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][0] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (LEFTOFFSET, TOPOFFSET), (LEFTOFFSET + SIDELENGTH, TOPOFFSET+SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[TOPOFFSET: TOPOFFSET + SIDELENGTH, CENTERWIDTH: CENTERWIDTH + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][1] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (CENTERWIDTH, TOPOFFSET), (CENTERWIDTH + SIDELENGTH, TOPOFFSET + SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[TOPOFFSET: TOPOFFSET + SIDELENGTH, RIGHTOFFSET: RIGHTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][2] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (RIGHTOFFSET, TOPOFFSET), (RIGHTOFFSET + SIDELENGTH, TOPOFFSET + SIDELENGTH),
                      (0, 255, 0), 2)

    #   Middle
    if getContours(imgCanny[CENTERHEIGHT: CENTERHEIGHT + SIDELENGTH, LEFTOFFSET: LEFTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][3] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (LEFTOFFSET, CENTERHEIGHT), (LEFTOFFSET + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[CENTERHEIGHT: CENTERHEIGHT + SIDELENGTH, CENTERWIDTH: CENTERWIDTH + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][4] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (CENTERWIDTH, CENTERHEIGHT), (CENTERWIDTH + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[CENTERHEIGHT: CENTERHEIGHT + SIDELENGTH, RIGHTOFFSET: RIGHTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][5] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (RIGHTOFFSET, CENTERHEIGHT), (RIGHTOFFSET + SIDELENGTH, CENTERHEIGHT + SIDELENGTH),
                      (0, 255, 0), 2)

    #   Bottom
    if getContours(imgCanny[BOTTOMOFFSET: BOTTOMOFFSET + SIDELENGTH, LEFTOFFSET: LEFTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][6] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (LEFTOFFSET, BOTTOMOFFSET), (LEFTOFFSET + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[BOTTOMOFFSET: BOTTOMOFFSET + SIDELENGTH, CENTERWIDTH: CENTERWIDTH + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][7] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (CENTERWIDTH, BOTTOMOFFSET), (CENTERWIDTH + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                      (0, 255, 0), 2)

    if getContours(imgCanny[BOTTOMOFFSET: BOTTOMOFFSET + SIDELENGTH, RIGHTOFFSET: RIGHTOFFSET + SIDELENGTH]):
        #print(SquareColourNames[colourIndex - 1])
        FaceArray[faceIndex][8] = SquareColourNames[colourIndex - 1]
        cv2.rectangle(img, (RIGHTOFFSET, BOTTOMOFFSET), (RIGHTOFFSET + SIDELENGTH, BOTTOMOFFSET + SIDELENGTH),
                      (0, 255, 0), 2)



    # img Text

    # Display Expected Face and any messages
    cv2.putText(img, "Face: " + str(SquareColourNames[faceIndex]), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 150, 0),
                1)
    cv2.putText(img, "Message: " + displayMessage, (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 150, 0),1)

    # Display Current input Faces
    cv2.putText(img, str(FaceArray[0]), (5, 100), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)
    cv2.putText(img, str(FaceArray[1]), (5, 115), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)
    cv2.putText(img, str(FaceArray[2]), (5, 130), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)
    cv2.putText(img, str(FaceArray[3]), (5, 145), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)
    cv2.putText(img, str(FaceArray[4]), (5, 160), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)
    cv2.putText(img, str(FaceArray[5]), (5, 175), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 150, 0), 1)

    # Display Detected colour in ROI
    cv2.putText(img, str(FaceArray[faceIndex][0]), (LEFTOFFSET, TOPOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][1]), (CENTERWIDTH, TOPOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][2]), (RIGHTOFFSET, TOPOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, str(FaceArray[faceIndex][3]), (LEFTOFFSET, CENTERHEIGHT + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][4]), (CENTERWIDTH, CENTERHEIGHT + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][5]), (RIGHTOFFSET, CENTERHEIGHT + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, str(FaceArray[faceIndex][6]), (LEFTOFFSET, BOTTOMOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][7]), (CENTERWIDTH, BOTTOMOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(FaceArray[faceIndex][8]), (RIGHTOFFSET, BOTTOMOFFSET + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    # If cubeSolution not "" Display Text
    if cubeSolution != "":
        cv2.putText(img, "Solution: ", (5,445), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 150, 0), 1)
        cv2.putText(img, str(cubeSolution), (5,460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 150, 0), 1)

    # Display Images
    cv2.imshow("img", img)
    cv2.imshow("Blur", imgBlur)
    cv2.imshow("Canny", imgCanny)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)

    # At the end of the loop increment colourIndex
    # Therefor New threshold next loop
    # Reset threshold at the end of the array
    colourIndex = colourIndex + 1
    if colourIndex >= len(SquareColourValues): colourIndex = 0

    # To progress through the Faces press "w"
    # .waitkey(delay) delay = time in ms
    if cv2.waitKey(WAITKEYDELAY) & 0xFF == ord("w"):

        # If the Centre Tile of the rubik == Expected Colour
        if SquareColourNames[faceIndex] == FaceArray[faceIndex][4]:

            # If faceIndex is less than 5. Not all faces input
            if faceIndex <= 4:
                # Increment faceIndex to move onto the next Face
                faceIndex = faceIndex + 1
                # Reset the Displayed message as the program should be functioning
                # As intended if faces are being entered
                displayMessage = ""

            # else if all face have been entered
            elif faceIndex == 5:

                #Y, B, R, G, O, W

                # Set Face configs to string
                yellowString = "".join(YellowFace)
                blueString = "".join(BlueFace)
                redString = "".join(RedFace)
                greenString = "".join(GreenFace)
                orangeString = "".join(OrangeFace)
                whiteString = "".join(WhiteFace)

                # Concatinate face string
                cubeString = yellowString + blueString + redString + greenString + orangeString + whiteString
                cube = cubeString

                #   displayMessage wont update on image until next frame, after the solve.
                displayMessage = "Solving..."
                print(displayMessage)

                # Attempt to solve
                try:
                    # .solve takes arguments, string, solving algorithm
                    cubeSolution = utils.solve(cube, 'Kociemba')

                # if the solve fails, inform the user
                except:
                    displayMessage = "Cube Solver error"
                    print(displayMessage)
                    print("Cube String: " + cubeString)

                # if the solve does not fail, inform the user
                else:
                    displayMessage = "Solution found"
                    print(displayMessage)
                    print("Solution: " + str(cubeSolution))
                    print("Cube String: " + cubeString)

        # If the Centre Tile of the rubik != Expected Colour
        else:
            displayMessage = "Centre Square does not match the current Face"
            print(displayMessage)


    # Navigate one step back, reset current face before moving
    if cv2.waitKey(WAITKEYDELAY) & 0xFF == ord("e"):

        # If the current face is not the first face
        if faceIndex >= 1:
            # Set current face array elements and cubeSolution to ""
            FaceArray[faceIndex][:] = ["", "", "", "", "", "", "", "", ""]
            cubeSolution = ""
            # Step back faceIndex
            faceIndex = faceIndex - 1
            # Alert the user to action
            displayMessage = "Moving back to " + SquareColourNames[faceIndex]
            print(displayMessage)

        # If the current face is the first face, no place to move back to
        else:
            displayMessage = "At Starting Face "
            print(displayMessage)



    # Reset all Faces, start from first face
    if cv2.waitKey(WAITKEYDELAY) & 0xFF == ord("r"):

        # Reset all properties, message, solution
        print("Reset")
        displayMessage = ""
        cubeSolution = ""
        # Loop for Faces in FaceArray, Set all Elements to ""
        for face in FaceArray:
            face[:] = ["", "", "", "", "", "", "", "", ""]
        # Set faceIndex to 0
        faceIndex = 0
        cubeSolution = ""

    # Allow User to quit
    if cv2.waitKey(WAITKEYDELAY) & 0xFF == ord("q"):
        break
