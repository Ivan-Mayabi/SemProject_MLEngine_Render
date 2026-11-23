from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from risk_engine import OutsideCoverage, RiskEngine

HERE = Path(__file__).parent
app = FastAPI(title="MOS risk server")
engine = RiskEngine.load(HERE / "risk_model.joblib", HERE / "data" / "nairobi_cells.csv")

class PredictRequest(BaseModel):
    coordinates: list[list[float]] = Field(min_length=1)
    mode: str = "MATATU"
    hour: int = Field(ge=0, le=23)

class PredictResponse(BaseModel):
    overall: float
    accident: float
    flooding: float
    lighting: float

@app.get("/health")
def health():
    return {"status": "ok", "cells": len(engine.cells)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, debug: bool = Query(False)):
    try:
        out = engine.predict(req.coordinates, req.mode, req.hour, debug=debug)
    except (OutsideCoverage, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return JSONResponse(out) if debug else out