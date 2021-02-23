import cv2
import numpy as np

#burada kamera ise 0
#video ise tam adresi

cap = cv2.VideoCapture('C:/YOLO/yolo_pretrained_video/videos/people.mp4')

while True:
    ret, frame = cap.read()
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)

    labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
         "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
         "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
         "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
         "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
         "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
         "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
         "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
         "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
         "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


    model = cv2.dnn.readNetFromDarknet("C:\YOLO\pretrained_model\yolov3.cfg","C:\YOLO\pretrained_model\yolov3.weights") #burada cfg ve weights dosyalarını koyuyoruz tam adres önemli

    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)


    ############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################
    
    ids_list = []
    boxes_list = []
    confidences_list = []
    
    ############################ END OF OPERATION 1 ########################
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.30:
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                
                ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                ############################ END OF OPERATION 2 ########################
                
                
                
    ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################
                
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
         
    for max_id in max_ids:
        
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
         
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
      
    ############################ END OF OPERATION 3 ########################
    
        label = "{}: {:.2f}%".format(label, confidence*100)
        print("predicted object {}".format(label))

    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
            
cap.release()
cv2.destroyAllWindows()
    
