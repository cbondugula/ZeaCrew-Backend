from fastapi import FastAPI, HTTPException, Query, APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv
from .llms_fastapi import get_current_user  # Import authentication dependency

load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["agent_db"]
integrations_collection = db["integrations"]

router = APIRouter()

@router.get("/integrations")
async def get_integrations(user: dict = Depends(get_current_user)):
    try:
        integrations = list(integrations_collection.find({}, {"_id": 1, "name": 1, "auth_method": 1, "created_on": 1}))
        for integration in integrations:
            integration["_id"] = str(integration["_id"])  # Convert ObjectId to string
            if "created_on" in integration and isinstance(integration["created_on"], datetime):
                integration["created_on"] = integration["created_on"].isoformat()  # Convert datetime to string
        return JSONResponse(content={"success": True, "integrations": integrations}, status_code=200)
    except Exception as e:
        print(f"Error fetching integrations: {e}")
        return JSONResponse(content={"success": False, "error": f"An error occurred: {e}"}, status_code=500)
# Add Integration
@router.post("/integrations")
async def add_integration(request: Request, user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        name = data.get("name")
        auth_method = data.get("auth_method")
        credentials = data.get("credentials")

        if not name or not auth_method or not credentials:
            return JSONResponse(content={"success": False, "error": "Missing required fields"}, status_code=400)

        integration = {
            "name": name,
            "auth_method": auth_method,
            "credentials": credentials,
            "created_on": datetime.utcnow(),
        }
        result = integrations_collection.insert_one(integration)
        return JSONResponse(content={"success": True, "message": "Integration added successfully", "id": str(result.inserted_id)}, status_code=201)
    except Exception as e:
        print(f"Error adding integration: {e}")
        return JSONResponse(content={"success": False, "error": "An error occurred."}, status_code=500)

# Edit Integration
@router.put("/integrations/{integration_id}")
async def edit_integration(integration_id: str, request: Request, user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        update_data = {}
        
        if "name" in data:
            update_data["name"] = data["name"]
        if "auth_method" in data:
            update_data["auth_method"] = data["auth_method"]
        if "credentials" in data:
            update_data["credentials"] = data["credentials"]
        
        if not update_data:
            return JSONResponse(content={"success": False, "error": "No valid fields to update"}, status_code=400)
        
        result = integrations_collection.update_one({"_id": ObjectId(integration_id)}, {"$set": update_data})
        
        if result.matched_count == 0:
            return JSONResponse(content={"success": False, "error": "Integration not found"}, status_code=404)
        
        return JSONResponse(content={"success": True, "message": "Integration updated successfully"}, status_code=200)
    except Exception as e:
        print(f"Error editing integration: {e}")
        return JSONResponse(content={"success": False, "error": "An error occurred."}, status_code=500)

# Delete Integration
@router.delete("/integrations/{integration_id}")
async def delete_integration(integration_id: str, user: dict = Depends(get_current_user)):
    try:
        result = integrations_collection.delete_one({"_id": ObjectId(integration_id)})
        
        if result.deleted_count == 0:
            return JSONResponse(content={"success": False, "error": "Integration not found"}, status_code=404)
        
        return JSONResponse(content={"success": True, "message": "Integration deleted successfully"}, status_code=200)
    except Exception as e:
        print(f"Error deleting integration: {e}")
        return JSONResponse(content={"success": False, "error": "An error occurred."}, status_code=500)
