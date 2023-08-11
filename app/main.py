import tensorflow as tf, urllib.request, numpy as np, requests, time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "heic"]

class ImageRequest(BaseModel):
    images: List[str]
    microscope_images: List[str]
    data: dict

@app.on_event("startup")
def load_model():
    global classNames
    classNames = ["Coal-Ash_Corrosion (FC-CA)", "Dissimilar_Metal_Weld (SR-DM)", "Erosion_Damage", "FAC", 
                  "Fireside_Corrosion", "Fly_Ash_Erosion (ER-FA)", "Long_Term_Overheat_Damage", "Long_Term_Overheating (SR-LT)", 
                  "Oxygen_Corrosion", "Short_Term_Overheat_Damage", "Soot_Blower_Erosion (ER-SB)", "Welding_Defects"]

@app.post("/predictFromImage")
async def predictFromImage(files: List[UploadFile] = File(...)):
    predictions = []
    candidates = []
    
    try:
        for file in files:
            startProcessTime = time.process_time()
            
            fileExtension = file.filename.split(".")[-1].lower()
            
            if fileExtension not in ALLOWED_EXTENSIONS:
                return {"error": f"File extension '.{fileExtension}' not allowed."}
            
            imageContent = await file.read()
            
            image = tf.image.decode_image(imageContent, channels = 3)
            image = tf.image.resize(image, [299, 299])
            imageArray = tf.expand_dims(image, axis = 0)
            imageArray = imageArray / 255.0
            
            serverURL = "http://10.31.1.158:8501/v1/models/model:predict"
            inputData = {"instances": imageArray.numpy().tolist()}
            response = requests.post(serverURL, json = inputData)
            
            prediction = response.json()["predictions"][0]
            predictions.append(prediction)
            
            result = []
            for classIndex, className in enumerate(classNames):
                confidence = prediction[classIndex]
                confidenceStr = f"{confidence:.4f}"
                result.append({"Class": className, "Confidence": float(confidenceStr)})
            
            endProcessTime = time.process_time()
            processTime = round(endProcessTime - startProcessTime, 4)
            
            candidates.append({"Filename": file.filename, "Result": result, "ProcessTime": processTime})
        
        avgPrediction = np.mean(predictions, axis=0)
        
        finalAnswer = []
        for classIndex, className in enumerate(classNames):
            confidence = avgPrediction[classIndex]
            confidenceStr = f"{confidence:.4f}"
            finalAnswer.append({"Class": className, "Confidence": float(confidenceStr)})
        
        totalProcessTime = round(sum(candidate["ProcessTime"] for candidate in candidates), 4)
        
        return {
            "FinalAnswer": finalAnswer,
            "TotalProcessTime": totalProcessTime,
            "Candidates": candidates,
        }
    except urllib.error.URLError as e:
        print("Error:", e)
        raise HTTPException(status_code = 500, detail="Internal Server Error")

@app.post("/predictFromPath")
async def predictFromPath(item: ImageRequest):
    images = item.images
    microscopeImages = item.microscope_images
    data = item.data
    
    predictions = []
    candidates = []
    
    try:
        for path in images:
            startProcessTime = time.process_time()
            with urllib.request.urlopen(path) as response:
                imgData = response.read()
            
            imgBuffer = BytesIO(imgData)
            imageContent = imgBuffer.getvalue()
            
            image = tf.image.decode_image(imageContent, channels = 3)
            image = tf.image.resize(image, [299, 299])
            imageArray = tf.expand_dims(image, axis = 0)
            imageArray = imageArray / 255.0
            
            serverURL = "http://10.31.1.158:8501/v1/models/model:predict"
            inputData = {"instances": imageArray.numpy().tolist()}
            response = requests.post(serverURL, json = inputData)
            
            prediction = response.json()["predictions"][0]
            predictions.append(prediction)
            
            result = []
            for classIndex, className in enumerate(classNames):
                confidence = prediction[classIndex]
                confidenceStr = f"{confidence:.4f}"
                result.append({"Class": className, "Confidence": float(confidenceStr)})
            
            endProcessTime = time.process_time()
            processTime = round(endProcessTime - startProcessTime, 4)
            
            candidates.append({"ImageUrl": path, "Result": result, "ProcessTime": processTime})
        avgPrediction = np.mean(predictions, axis=0)
        
        finalAnswer = []
        classMapping = {
            "Coal-Ash_Corrosion (FC-CA)": "Corrosion",
            "Fireside_Corrosion": "Corrosion",
            "Oxygen_Corrosion": "Corrosion",
            "Fly_Ash_Erosion (ER-FA)": "Erosion",
            "Soot_Blower_Erosion (ER-SB)": "Erosion",
            "Erosion_Damage": "Erosion",
            "FAC": "FAC",
            "Short_Term_Overheat_Damage": "STO",
            "Long_Term_Overheat_Damage": "LTO",
            "Long_Term_Overheating (SR-LT)": "LTO",
            "Dissimilar_Metal_Weld (SR-DM)": "Welding",
            "Welding_Defects": "Welding",
        }
        
        combinedClasses = {}
        for classIndex, className in enumerate(classNames):
            if className in classMapping:
                finalClassName = classMapping[className]
                if finalClassName not in combinedClasses:
                    combinedClasses[finalClassName] = []
                combinedClasses[finalClassName].append(avgPrediction[classIndex])
            else:
                finalClassName = className
                combinedClasses[finalClassName] = [avgPrediction[classIndex]]
        
        for finalClassName, classConfidences in combinedClasses.items():
            avgConfidence = np.sum(classConfidences)
            confidenceStr = f"{avgConfidence:.4f}"
            finalAnswer.append({"Class": finalClassName, "Confidence": float(confidenceStr)})
        
        totalProcessTime = round(sum(candidate["ProcessTime"] for candidate in candidates), 4)
        
        return {
            "FinalAnswer": finalAnswer,
            "TotalProcessTime": totalProcessTime,
            "Candidates": candidates,
        }
    except urllib.error.URLError as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")