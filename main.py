from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import polars as pl
import io
import json

from cleaner import evaluate_dataset, execute_flexible_clean

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_STORE = {}
RAW_BAK_STORE = {} # Holds original unmodified snapshots
AUDIT_TRAILS = {}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV datasets are accepted.")
    
    contents = await file.read()
    try:
        df = pl.read_csv(io.BytesIO(contents))
        session_id = file.filename
        
        DATA_STORE[session_id] = df
        RAW_BAK_STORE[session_id] = df.clone() # Keep clean backup snapshot
        AUDIT_TRAILS[session_id] = []
        
        profile = evaluate_dataset(df)
        
        # Take preview slice sample of first 10 rows safely mapped to string matrices
        preview_data = df.head(10).to_dicts()
        
        return {
            "session_id": session_id,
            "summary": profile["summary"],
            "columns": profile["columns"],
            "preview": preview_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

class CleaningConfig(BaseModel):
    deduplicate: bool
    columns: dict

@app.post("/api/clean/execute")
async def execute_clean(
    session_id: str = Query(...),
    config: CleaningConfig = Body(...) # FIXED: Changed from Query string to structured body JSON
):
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Active dataset session not found.")
        
    try:
        # Convert Pydantic object back into standard dictionary
        parsed_config = config.model_dump()
        
        # Always clean from our original backup to avoid multi-run compound bias
        base_df = RAW_BAK_STORE[session_id].clone()
        
        df_cleaned, ledger = execute_flexible_clean(base_df, parsed_config)
        
        DATA_STORE[session_id] = df_cleaned
        AUDIT_TRAILS[session_id] = ledger
        
        updated_profile = evaluate_dataset(df_cleaned)
        
        return {
            "status": "success",
            "audit_ledger": ledger,
            "updated_summary": updated_profile["summary"],
            "updated_columns": updated_profile["columns"],
            "preview": df_cleaned.head(10).to_dicts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation execution failed: {str(e)}")
    
@app.get("/api/export/csv")
async def export_csv(session_id: str = Query(...)):
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    df = DATA_STORE[session_id]
    buffer = io.BytesIO()
    df.write_csv(buffer)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=clean_{session_id}"})