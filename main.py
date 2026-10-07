from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="CaliPredict")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
classifier = joblib.load("california.joblib")


@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/predict")
def predict(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
):
    input_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude
    }])

    prediction = classifier["Model"].predict(input_data)

    return {"prediction": float(prediction[0])}


@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    df_test = pd.read_csv(file.file)
    prediction = classifier["Model"].predict(df_test)
    return {"predictions": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
