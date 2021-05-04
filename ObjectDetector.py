import cv2

img = cv2.imread("../Object_detection/Data/TrafficClearance.jpeg")    ## Reading the Image file
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRBG)                           ## Color Convertion will be nessesary to display output using matplotlib
                                                                      ##Cause CV2 reads images as BGR



nameFile = "../Object_detection/coco.names"                           ## Reading the coco names file - dataset
                                                                      ## this is the file that consists of names of objects to be detected
    
with open(nameFile,"rt") as f:
    class_names = f.read().rstrip("\n").split("\n")
    print(class_names)

    


configPath = "../Object_detection/ssd/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"         ## Assigning the ssd file as ConfigPath
weightsPath = "../Object_detection/frozen_inference_graph.pb"                               ## Assigning the protobuf file as weightsPath


    
net = cv2.dnn_DetectionModel(weightsPath,configPath)                                        ## Building the dnn model
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



classIds, confids, bbox = net.detect(img,0.5)                                              
print(classIds,bbox)                                                ## Using the model to detect the objects, this line returns the 
                                                                    ## classIds --> Index position of detected objects corresponding to the names file/classes.
                                                                    ## confids --> Confidence/Detection accuracy
                                                                    ## bbox --> coordinates to draw box around the detected image
            
            
            
    
if len(classIds) != 0:                                              ## Checking if classIds available in the list, this is done to avoid error crashing
    
    for classId, confidence, box in zip(classIds.flatten(),confids.flatten(),bbox):
        
        cv2.rectangle(img,box,color = (252, 255, 173),thickness=2)
        ## drawing the rectange
        
        cv2.putText(img,class_names[classId-1],org = (box[0]+10,box[1]+30),fontFace = cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(252, 255, 173),thickness=2)
        ## Adding the Object Name to the output window/image using the classId 
        
        #cv2.putText(img,str(round(confidence*100,2)),(box[0]+50,box[1]+30),cv2.FONT_HERSHEY_PLAIN,1,(252, 255,173),2)
        ## If confidence score has to be view in the output window/image
        
        
cv2.imshow("frame",img)                                          ## Displaying the output Window



while (1):
    if cv2.waitKey(10) &  0xFF==27:                             ## to break out of the loop and stop displaying the image
        break

cv2.destroyAllWindows()