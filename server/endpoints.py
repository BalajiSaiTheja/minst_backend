import uvicorn
from fastapi import FastAPI,File,UploadFile,Response
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from modules.Predict import Predict
from modules.visualizer import Visualize

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods =["*"],
    allow_headers=["*"] 
)


@app.post("/predict")
async def predict(uploaded_image:UploadFile=File(...)):
    contents = await uploaded_image.read()
    
    try:
        pred = Predict.PredictFromBytes(contents)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    print(f"received image :  {uploaded_image.filename}")

    return {
        "filename":uploaded_image.filename,
        "status":"success",
        "prediction":pred,
        "message":"Image received and processed"
    }


@app.post("/visualize")
async def visualize(uploaded_image:UploadFile=File(...)):
    contents = await uploaded_image.read()
    print("visualize")
    try:
        pred = Visualize.plot_layers(contents,buffer=True)
        
    except Exception as e:
        print(f"Visualization Error: {e}")
        return Response(content=str(e), status_code=500)

    print(f"received image :  {uploaded_image.filename}")

    return Response(content=pred,media_type="image/png")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)