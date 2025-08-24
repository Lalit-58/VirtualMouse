import numpy as np
import webb as htm
import time
import autopy
import cv2
import pyautogui  # Import pyautogui for scrolling

# Camera and screen setup
wCam, hCam = 1280, 720
frameR = 50
smoothening = 3

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
scrollThreshold = 10  # Threshold for fist movement detection
scrollSpeed = 800 # Adjust this to change the scroll speed

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
wScr, hScr = autopy.screen.size()

# Variable to store the last y position of the fist when it is closed
lastFistY = 0
scrolling = False  # Flag to indicate if scrolling is active

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]   # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip
            x3, y3 = lmList[4][1:]   # Thumb tip

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            # Only index finger up => Move mode
            if fingers[1] == 1 and fingers[2] == 0:
                xMapped = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                yMapped = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                clocX = plocX + (xMapped - plocX) / smoothening
                clocY = plocY + (yMapped - plocY) / smoothening

                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Index and middle up => Left click 
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

            # Index and thumb up, close => Right click 
            if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0:
                length, img, lineInfo = detector.findDistance(4, 8, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)

            # Detect closed fist (all fingers down)
            if all(finger == 0 for finger in fingers):  # Fist is closed
                # Track fist vertical movement for scrolling
                if lastFistY != 0:
                    yMovement = y1 - lastFistY  # Vertical movement of fist
                    if yMovement < -scrollThreshold:  # Fist moving upward (scroll up)
                        if not scrolling:
                            pyautogui.scroll(scrollSpeed)  # Scroll up
                            scrolling = True
                    elif yMovement > scrollThreshold:  # Fist moving downward (scroll down)
                        if not scrolling:
                            pyautogui.scroll(-scrollSpeed)  # Scroll down
                            scrolling = True
                    else:
                        # If fist stops moving significantly, stop scrolling
                        scrolling = False

                # Update the last fist y position
                lastFistY = y1

        # FPS display
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
