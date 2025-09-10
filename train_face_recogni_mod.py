import cv2 
import os 
import numpy as np


face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
actors_name = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence', 
               'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 
               'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith'] 


path = r"C:\Users\bca19\OneDrive\Desktop\my datasets for project\Celebrity Faces Dataset"

lables = []
actor_face = []


for actors in actors_name:
    actors_folder = os.path.join(path, actors) 
    actor_index = actors_name.index(actors)

    for images in os.listdir(actors_folder):
        print(images)
        actor_img_path = os.path.join(actors_folder,images) 
        array_img = cv2.imread(actor_img_path) 
        grey_image = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
        
        face_roi  = face_dector.detectMultiScale(grey_image, 1.2,3) 
        
        for x,y,w,h in face_roi:
           
            crop_face =  grey_image[y : y+h, x: x+w]
            
            lables.append(actor_index)
            actor_face.append(crop_face )
            
            
lables_array = np.array(lables) 
actor_face_array = np.array(actor_face, dtype = 'object') 

# model----------> Algo

model = cv2.face.LBPHFaceRecognizer_create() 

model.train(actor_face_array, lables_array) 

model.save("face_recognize_system.yml")
        
        