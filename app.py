from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import uvicorn

app = FastAPI()
session = ort.InferenceSession("model.onnx")

# Define input format
class ModelInput(BaseModel):
    data: list[list[float]]  # Expecting a 2D list (batch_size, sequence_length)
    
@app.post("/predict/")
async def predict(input_data: ModelInput):
    input_array = np.array(input_data.data, dtype=np.float32).reshape(-1, 100, 1)  # Ensure correct shape
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})

    return {"predictions": output[0].tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
