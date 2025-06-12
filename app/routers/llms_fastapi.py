from fastapi import FastAPI, HTTPException, Query, APIRouter, Request,Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import jwt
import bcrypt
import os
from dotenv import load_dotenv
import uvicorn
load_dotenv()
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr, ValidationError, validator,Field



# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("JWT_SECRET", "your_secret_key")  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

client = MongoClient(MONGO_URI)
db = client["agent_db"]
users_collection = db["users"]
llm_connections_collection = db["llm_connections"]
llm_models_collection = db["llm_models"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/signin")

router = APIRouter()

ALLOWED_PROVIDERS = [
    "OpenAI", "Custom-openai-compatible", "Anthropic", "Groq", "Azure",
    "Cohere", "Gemini", "Bedrock", "Cerebras", "Internal_openai","Grok","IBM"
]
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        return user_id  # Return only user_id if that's enough for you

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )

    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


import re

class SignupSchema(BaseModel):
    email: EmailStr=Field(..., min_length=6, max_length=100)
    password: str

    @validator("password")
    def validate_password_strength(cls, password):
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", password):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            raise ValueError("Password must contain at least one special character")
        return password
    

class SigninSchema(BaseModel):
    email: EmailStr
    password: str
from fastapi.encoders import jsonable_encoder
#Routes
@router.post("/auth/signup")
async def signup(request: Request):
    try:
        data = SignupSchema(**await request.json())
    except ValidationError as e:
        errors = []
        for err in e.errors():
            errors.append({
                "field": err.get("loc")[0] if err.get("loc") else "unknown",
                "message": err.get("msg")
            })
        return JSONResponse(
            content={"success": False, "errors": errors},
            status_code=422
        )

    except Exception as e:
        print("Unhandled exception during signup:", str(e))
        return JSONResponse(
            content={"success": False, "detail": "Internal server errors."},
            status_code=500
        )

    if users_collection.find_one({"email": data.email}):
        return JSONResponse(
            content={"success": False, "error": "Email already registered"}, 
            status_code=400
        )

    hashed_password = hash_password(data.password)
    user = {
        "email": data.email,
        "password": hashed_password,
        "created_on": datetime.utcnow()
    }
    user_id = users_collection.insert_one(user).inserted_id

    return JSONResponse(
        content={"success": True, "message": "User created successfully", "user_id": str(user_id)}, 
        status_code=201
    )


@router.post("/auth/signin")
async def signin(request: Request):
    try:
        data = SigninSchema(**await request.json())
    except ValidationError as e:
        return JSONResponse(content={"success": False, "error": e.errors()}, status_code=422)

    user = users_collection.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password"]):
        return JSONResponse(content={"success": False, "error": "Invalid email or password"}, status_code=401)

    access_token = create_access_token(
        {"user_id": str(user["_id"])},
        timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )

    return JSONResponse(content={"success": True, "access_token": access_token, "token_type": "bearer"}, status_code=200)

# @router.post("/auth/signup")
# async def signup(request: Request):
#     data = await request.json()
#     email = data.get("email")
#     password = data.get("password")
#     if not email or not password:
#         return JSONResponse(content={"success": False,"error": "Email and password required"}, status_code=400)
    
#     existing_user = users_collection.find_one({"email": email})
#     if existing_user:
#         return JSONResponse(content={"success": False,"error": "Email already registered"}, status_code=400)
    
#     hashed_password = hash_password(password)
#     user = {"email": email, "password": hashed_password, "created_on": datetime.utcnow()}
#     user_id = users_collection.insert_one(user).inserted_id
#     return JSONResponse(content={"success": True,"message": "User created successfully", "user_id": str(user_id)}, status_code=201)

# @router.post("/auth/signin")
# async def signin(request: Request):
#     data = await request.json()
#     email = data.get("email")
#     password = data.get("password")
#     if not email or not password:
#         return JSONResponse(content={"success": False,"error": "Email and password required"}, status_code=400)
    
#     user = users_collection.find_one({"email": email})
#     if not user or not verify_password(password, user["password"]):
#         return JSONResponse(content={"success": False,"error": "Invalid email or password"}, status_code=401)
    
#     access_token = create_access_token({"user_id": str(user["_id"])}, timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
#     return JSONResponse(content={"success": True,"access_token": access_token, "token_type": "bearer"}, status_code=200)



@router.post("/add_llm_connection")
async def add_llm_connection(request: Request, user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        name = data.get("name")
        provider = data.get("provider")  # Renamed from provider
        env_variables = data.get("envVariables", {})
        models = data.get("Models", [])

        if not name or not provider:
            raise HTTPException(status_code=400, detail="Missing required fields: 'name' or 'id'")

        if not isinstance(env_variables, dict) or not isinstance(models, list):
            raise HTTPException(status_code=400, detail="Invalid data format for 'envVariables' or 'Models'")

        # user_data = user.get("user", {})
        user_id = user

        # print("Extracted user:", user_data)  # Debugging
        # print("Extracted user_id:", user_id, type(user_id))  # Debugging

        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized: User not found")

        # Convert ObjectId to string if needed
        user_id = str(user_id)

        connection_entry = {
            "user_id": user_id,
            "name": name,
            "provider": provider,  # Updated key name from provider
            "envVariables": env_variables,
            "Models": models,
            "createdOn": datetime.utcnow(),
            "updatedOn": datetime.utcnow()
        }

        inserted_id = llm_connections_collection.insert_one(connection_entry).inserted_id
        return JSONResponse(
            content={
                "success": True,
                "connection_id": str(inserted_id),
                "message": "LLM Connection added successfully"
            },
            status_code=201
        )
    except HTTPException as http_err:
        return JSONResponse(content={"success": False, "error": http_err.detail}, status_code=http_err.status_code)
    except Exception as e:
        # logger.error(f"Error adding LLM Connection: {str(e)}")
        return JSONResponse(content={"success": False, "error": "An internal server error occurred."}, status_code=500)




@router.get("/get_all_llm_connections")
async def get_all_llm_connections(user: dict = Depends(get_current_user)):
    try:
        # user_data = user.get("user", {})
        # user_id = user_data.get("_id")
        user_id = user
        if not user_id:
            return JSONResponse(content={"error": "Unauthorized: User ID not found"}, status_code=401)

        user_id = str(user_id)  # Ensure user_id is a string

        query = {"user_id": user_id}
        print("Executing Query:", query)

        connections = list(llm_connections_collection.find(query))
        print("Found Connections:", connections)
        
        if not connections:
            return JSONResponse(content={"success": False, "connections": [], "error": "No LLM connections found"}, status_code=404)

        # Convert ObjectId fields to strings
        # Convert BSON to JSON serializable format
        for conn in connections:
            conn["_id"] = str(conn["_id"])  # Convert ObjectId
            if "createdOn" in conn:
                conn["createdOn"] = conn["createdOn"].isoformat()
            if "updatedOn" in conn:
                conn["updatedOn"] = conn["updatedOn"].isoformat()

        print(f"Connections found: {connections}")  # Debugging output
        return JSONResponse(content={"success": True, "connections": connections}, status_code=200)

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging
        return JSONResponse(content={"success": False, "error": f"An error occurred: {str(e)}"}, status_code=500)

# 
@router.put("/edit_llm_connection")
async def edit_llm_connection(request: Request, user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        connection_id = data.get("_id")

        if not connection_id:
            return JSONResponse(content={"success": False, "error": "Missing '_id' in request body"}, status_code=400)

        # Extract user_id from nested user structure
        # user_data = user.get("user", {})  # Get inner user object
        user_id = user

        if not user_id:
            print(f"Error: Missing '_id' in user object: {user}")  # Debugging
            raise HTTPException(status_code=401, detail="Unauthorized: User ID not found")

        user_id = str(user_id)  # Convert ObjectId to string

        # Ensure ObjectId conversion
        result = llm_connections_collection.update_one(
            {"_id": ObjectId(connection_id), "user_id": user_id},
            {"$set": {
                "name": data.get("name"),
                "provider": data.get("provider"),
                "envVariables": data.get("envVariables"),
                "openaiModels": data.get("openaiModels"),
                "updatedOn": datetime.utcnow(),
            }}
        )

        if result.matched_count == 0:
            return JSONResponse(content={"success": False, "error": "LLM Connection not found"}, status_code=404)

        return JSONResponse(content={"success": True, "message": "LLM Connection updated successfully"}, status_code=200)

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging
        return JSONResponse(content={"success": False, "error": f"An error occurred: {str(e)}"}, status_code=500)

# Delete LLM Connection (Restricted to User)
@router.delete("/delete_llm_connection/{connection_id}")
async def delete_llm_connection(connection_id: str, user: dict = Depends(get_current_user)):
    try:
        if not ObjectId.is_valid(connection_id):
            return JSONResponse(content={"success": False, "error": "Invalid connection ID"}, status_code=400)

        # Extract user_id from nested user structure
        # user_data = user.get("user", {})  # Get inner user object
        # user_id = user_data.get("_id")
        user_id=user

        if not user_id:
            print(f"Error: Missing '_id' in user object: {user}")  # Debugging
            raise HTTPException(status_code=401, detail="Unauthorized: User ID not found")

        user_id = str(user_id)  # Convert ObjectId to string

        # Check if the connection exists
        existing_connection = llm_connections_collection.find_one({"_id": ObjectId(connection_id), "user_id": user_id})
        print("Found Connection: ", existing_connection)  # Debugging

        if not existing_connection:
            return JSONResponse(content={"success": False, "error": "LLM Connection not found or unauthorized"}, status_code=404)

        # Proceed with deletion
        result = llm_connections_collection.delete_one({"_id": ObjectId(connection_id), "user_id": user_id})

        if result.deleted_count == 0:
            return JSONResponse(content={"success": False, "error": "Failed to delete LLM Connection"}, status_code=500)

        return JSONResponse(content={"success": True, "message": "LLM Connection deleted successfully"}, status_code=200)

    except Exception as e:
        print(f"Error in delete_llm_connection: {e}")  # Debugging
        return JSONResponse(content={"success": False, "error": f"An error occurred: {str(e)}"}, status_code=500)


@router.get("/get_allowed_providers")
async def get_allowed_providers(user: dict = Depends(get_current_user)):
    try:
        # Query the database to fetch all documents excluding 'models' field
        providers_data = llm_models_collection.find({}, {"models": 0})  # Excluding 'models' field
        
        # Convert the cursor to a list of dictionaries
        providers_list = list(providers_data)
        
        # Convert ObjectId to string for each document
        for provider in providers_list:
            provider["_id"] = str(provider["_id"])  # Convert _id to string
        
        # If no data is found
        if not providers_list:
            return JSONResponse(content={"success": False,"error": "No allowed providers found."}, status_code=404)

        # Return the fetched data
        return JSONResponse(content={"success": True,"providers": providers_list}, status_code=200)
    
    except Exception as e:
        print(f"Error in get_allowed_providers: {e}")
        return JSONResponse(content={"success": False,"error": f"An error occurred: {str(e)}"}, status_code=500)

@router.get("/get_llm_models")
async def get_llm_models(id: str = Query(None, description="Provider's _id"),user: dict = Depends(get_current_user)):
    try:
        if not id:
            return JSONResponse(content={"success": False,"error": "ID query parameter is required"}, status_code=400)

        # Convert the id string to ObjectId
        try:
            object_id = ObjectId(id)
        except Exception as e:
            return JSONResponse(content={"success": False,"error": "Invalid ID format"}, status_code=400)

        # Query the llm_models_collection to find models by _id
        llm_models = llm_models_collection.find_one({"_id": object_id}, {"models": 1, "_id": 0})  # Only fetching models field

        if not llm_models:
            return JSONResponse(content={"success": False,"error": f"No models found for _id: {id}"}, status_code=404)

        # If found, return the models field
        return JSONResponse(content={"success": True, "models": llm_models.get("models", [])}, status_code=200)

    except Exception as e:
        print(f"Error in get_llm_models: {e}")
        return JSONResponse(content={"success": False,"error": "An error occurred while fetching LLM models."}, status_code=500)

@router.post("/store_llm_models")
async def store_llm_models(request: Request):
    try:
        data = await request.json()
        if not data:
            return JSONResponse(content={"success": False,"error": "No data provided"}, status_code=400)

        provider = data.get("provider")
        models = data.get("models")

        if not provider or provider not in ALLOWED_PROVIDERS:
            return JSONResponse(content={"success": False,"error": "Invalid or missing 'provider'"}, status_code=400)

        if not models or not isinstance(models, list) or len(models) == 0:
            return JSONResponse(content={"success": False,"error": "Invalid or missing 'models'. Expected a non-empty list"}, status_code=400)

        llm_models_collection.update_one(
            {"provider": provider},
            {"$set": {"models": models, "updatedOn": datetime.now().isoformat()}},
            upsert=True
        )
        return JSONResponse(content={"success": True,"message": f"LLM Models stored successfully for {provider}"}, status_code=200)
    except Exception as e:
        print(f"Error in store_llm_models: {e}")
        return JSONResponse(content={"success": False,"error": "An error occurred."}, status_code=500)

# Edit LLM Models (PUT)
@router.put("/edit_llm_models/{provider}")
async def edit_llm_models(provider: str, request: Request):
    try:
        data = await request.json()
        if provider not in ALLOWED_PROVIDERS:
            return JSONResponse(content={"success": False,"error": "Invalid provider"}, status_code=400)
        
        update_data = {"updatedOn": datetime.now().isoformat()}
        
        if "models" in data and isinstance(data["models"], list):
            update_data["models"] = data["models"]
        
        result = llm_models_collection.update_one({"provider": provider}, {"$set": update_data})
        
        if result.matched_count == 0:
            return JSONResponse(content={"success": False,"error": "No models found for the provider"}, status_code=404)
        
        return JSONResponse(content={"success": True,"message": "LLM Models updated successfully"}, status_code=200)
    except Exception as e:
        print(f"Error in edit_llm_models: {e}")
        return JSONResponse(content={"success": False,"error": "An error occurred."}, status_code=500)
