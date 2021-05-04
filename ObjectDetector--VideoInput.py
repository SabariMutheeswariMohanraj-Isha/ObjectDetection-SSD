import cv2 

cap = cv2.VideoCapture("../Object_detection/Data/Walking-in-NYC-2x.mp4")  ## Reading the Video file
#cap = cv2.VideoCapture(0)                                                ## Instead, If we want to use webcam 
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

#cap.set(cv2.CAP_PROP_FPS,60)                                             ## re-assign the no of fps
#fps = int(cap.get(5))
#print("fps:", fps)                                                       ## To find the no of frames per second


nameFile = "../Object_detection/coco.names"                               ## Reading the coco names file - dataset
                                                                          ## this is the file that consists of names of objects to be detected

with open(nameFile,"rt") as f:
    class_names = f.read().rstrip("\n").split("\n")
    #print(class_names)
    
print(len(class_names))                                                  ## Return the no of names available - 80 classes


thresh = 0.6                                                             ## Assigning thresh which can be used later in the model



configPath = "../Object_detection/ssd/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  ## Assigning the ssd file as ConfigPath
weightsPath = "../Object_detection/frozen_inference_graph.pb"                        ## Assigning the protobuf file as weightsPath

## These will be available in OpenCV github, Tensorflow Object Detection API 



    
net = cv2.dnn_DetectionModel(weightsPath,configPath)                    ## Building the dnn model
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



while True:
    
    ret,frame = cap.read()                                             ## Reading the frames from the video captured
    
    #time.sleep(1/25)                                            ## importing time library, and use the sleep method to reduce the speed of the video, i.e. decrease frame rate
    
    
    
    classIds, confids, bbox = net.detect(frame,thresh)          ## Using the model to detect the objects, this line returns the 
                                                                ## classIds --> Index position of detected objects corresponding to the names file/classes.
                                                                ## confids --> Confidence/Detection accuracy
                                                                ## bbox --> coordinates to draw box around the detected image
    #print(classIds,bbox)

    
    
    if len(classIds) != 0:                                     ## Checking if classIds available in the list, this is done to avoid error crashing
        
        for classId, confidence, box in zip(classIds.flatten(),confids.flatten(),bbox):
            
            if (classId<=80):                                 ## Checking if the classId is the 80 classes, this is done to avoid list index out of range error 
                
                
                cv2.rectangle(frame,box,color = (252, 255, 173),thickness=2)
                ## drawing the rectange
                
                cv2.putText(frame,class_names[classId-1],org = (box[0]+10,box[1]+30),fontFace = cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(252, 255,173),thickness=2)
                ## Adding the Object Name to the output window/image using the classId 
                
                
                #cv2.putText(frame,str(round(confidence*100,2)),(box[0]+300,box[1]+30),cv2.FONT_HERSHEY_PLAIN,1,(252, 255,173),2)
                ## If confidence score has to be view in the output window/image

                cv2.imshow("Video Output",frame)              ## Displaying the output Window

    
    if cv2.waitKey(2) & 0xFF == 27:                          ## to break out of the loop and stop displaying the video
        break       

        
        
cap.release()        
cv2.destroyAllWindows()