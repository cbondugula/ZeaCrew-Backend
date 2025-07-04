from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse,Response
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from .llms_fastapi import get_current_user 

# Load environment variables
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["agent_db"]
tools_collection = db["tools"]

logging.basicConfig(level=logging.INFO)
# Initialize Router
router = APIRouter()

from pydantic import BaseModel
from typing import Optional
from datetime import datetime
ALLOWED_TOOLS = ["WebsiteSearchTool", "SerperDevTool"]

# class Tool(BaseModel):
#     name: str
#     api_key: str  # Ensure this field is explicitly defined

# class ToolInDB(Tool):
#     id: str
#     created_on: datetime

class Tool(BaseModel):
    name: str
    api_key: Optional[str] = None  # Optional for tools like WebsiteSearchTool
    website_url: Optional[str] = None      # Required for WebsiteSearchTool

class ToolInDB(Tool):
    id: str
    created_on: datetime

# @router.get("/tools")
# async def get_tools(user_id: dict = Depends(get_current_user)):
#     try:
#         # user_id = user.get("user")
#         tools_cursor = tools_collection.find({"user_id": user_id})
#         tools = await tools_cursor.to_list(length=None)

#         for tool in tools:
#             tool["id"] = str(tool["_id"])
#             del tool["_id"]

#             # Ensure API key is included if it exists
#             tool["api_key"] = tool.get("api_key", None)

#         return {"success": True, "tools": tools}

#     except Exception as e:
#         return {"success": False, "message": f"An error occurred: {str(e)}"}
@router.get("/tools")
async def get_tools(user_id: dict = Depends(get_current_user)):
    try:
        tools_cursor = tools_collection.find({"user_id": user_id})
        tools = await tools_cursor.to_list(length=None)

        for tool in tools:
            tool["id"] = str(tool["_id"])
            del tool["_id"]

            # Include relevant fields
            tool["api_key"] = tool.get("api_key")
            tool["url"] = tool.get("url")

        return {"success": True, "tools": tools}

    except Exception as e:
        return {"success": False, "message": f"An error occurred: {str(e)}"}


# POST /tools - Add a new tool
# @router.post("/tools", status_code=201)
# async def add_tool(tool: Tool, user_id: dict = Depends(get_current_user)):
#     try:
#         tool_dict = tool.dict()

#         # Ensure "api_key" is present
#         if not tool_dict.get("api_key"):
#             return {"success": False, "message": "API key is required."}

#         # Add timestamp
#         tool_dict["created_on"] = datetime.utcnow()
#         tool_dict["user_id"] = user_id

#         # Insert into DB
#         result = await tools_collection.insert_one(tool_dict)
#         tool_dict["id"] = str(result.inserted_id)

#         # Remove _id if present (just in case)
#         tool_dict.pop("_id", None)

#         return {"success": True, "tool": tool_dict}

#     except Exception as e:
#         return {"success": False, "message": f"An error occurred: {e}"}
# @router.post("/tools", status_code=201)
# async def add_tool(tool: Tool, user_id: dict = Depends(get_current_user)):
#     try:
#         tool_dict = tool.dict()

#         # Validate based on tool type
#         if tool_dict["name"] == "SerperDevTool" and not tool_dict.get("api_key"):
#             return {"success": False, "message": "API key is required for SerperDevTool."}
#         if tool_dict["name"] == "WebsiteSearchTool" and not tool_dict.get("website_url"):
#             return {"success": False, "message": "URL is required for WebsiteSearchTool."}
#         tool_dict["created_on"] = datetime.utcnow()
#         tool_dict["user_id"] = user_id

#         result = await tools_collection.insert_one(tool_dict)
#         tool_dict["id"] = str(result.inserted_id)
#         tool_dict.pop("_id", None)

#         return {"success": True, "tool": tool_dict}

#     except Exception as e:
#         return {"success": False, "message": f"An error occurred: {e}"}






# PUT /tools/{tool_id} - Edit an existing tool
@router.put("/tools/{tool_id}")
async def edit_tool(tool_id: str, tool: Tool, user_id: dict = Depends(get_current_user)):
    try:
        # user_id = user.get("user")

        existing_tool = await tools_collection.find_one({"_id": ObjectId(tool_id)})
        if not existing_tool:
            return {"success": False, "message": "Tool not found"}

        if str(existing_tool.get("user_id")) != str(user_id):
            raise HTTPException(status_code=403, detail="You are not authorized to update this tool")

        await tools_collection.update_one(
            {"_id": ObjectId(tool_id)},
            {"$set": tool.dict(exclude_unset=True)}
        )

        updated_tool = await tools_collection.find_one({"_id": ObjectId(tool_id)})
        updated_tool["id"] = str(updated_tool["_id"])
        del updated_tool["_id"]

        return {"success": True, "tool": updated_tool}

    except Exception as e:
        return {"success": False, "message": f"An error occurred: {str(e)}"}


# DELETE /tools/{tool_id} - Delete a tool
@router.delete("/tools/{tool_id}")
async def delete_tool(tool_id: str, user_id: dict = Depends(get_current_user)):
    try:
        # user_id = user.get("user")

        existing_tool = await tools_collection.find_one({"_id": ObjectId(tool_id)})
        if not existing_tool:
            return {"success": False, "message": "Tool not found"}

        if str(existing_tool.get("user_id")) != str(user_id):
            raise HTTPException(status_code=403, detail="You are not authorized to delete this tool")

        delete_result = await tools_collection.delete_one({"_id": ObjectId(tool_id)})

        return {
            "success": True,
            "message": "Tool deleted successfully",
            "tool_id": tool_id
        }

    except Exception as e:
        return {"success": False, "message": f"An error occurred: {str(e)}"}


ALLOWED_TOOLS = ["WebsiteSearchTool", "SerperDevTool"]
tools_collection1 = db["tools_dropdown"]
from pydantic import BaseModel, validator

ALLOWED_TOOLS = ["websearchtool", "serper"]

class Tool(BaseModel):
    id: str
    name: str
    key: Optional[str] = None
    type: str

    @validator("name")
    def validate_name(cls, v):
        if v.lower() not in ALLOWED_TOOLS:
            raise ValueError(f"Invalid tool name: {v}")
        return v.lower()

    @validator("key", always=True)
    def validate_key(cls, v, values):
        if values.get("name") == "serper" and not v:
            raise ValueError("Key is required for 'serper'")
        return v

@router.post("/store_tools_bulk")
async def store_tools_bulk(tools: List[Tool]):
    try:
        for tool in tools:
            update_data = {
                "id": tool.id,
                "name": tool.name,
                "type": tool.type,
                "updatedOn": datetime.utcnow().isoformat(),
            }
            if tool.key:
                update_data["key"] = tool.key

            await tools_collection1.update_one(
                {"id": tool.id},
                {"$set": update_data},
                upsert=True
            )

        return JSONResponse(content={"success": True, "message": "Tools stored successfully"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"success": False, "detail": str(e)}, status_code=500)
from typing import List, Dict  
@router.get("/get_toolsnames")
async def get_tools(user_id: str = Depends(get_current_user)):
    try:
        # Query all tools for the current user (no sort)
        cursor = tools_collection1.find({})
        tools: List[Dict] = await cursor.to_list(length=None)

        # Extract only required fields
        result = [
            {
                "id": tool.get("id"),
                "name": tool.get("name"),
                "type": tool.get("type"),
            }
            for tool in tools
        ]

        return JSONResponse(content={"success": True, "tools": result}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"Error fetching tools: {str(e)}"}, status_code=500)
# class Tool(BaseModel):
#     id: Optional[str] = None
#     name: str
#     # key: Optional[str] = None
#     api_key: Optional[str] = None
#     website_url: Optional[str] = None
#     type: str

# @router.post("/tools", status_code=201)
# async def add_tool(tool: Tool, user_id: str = Depends(get_current_user)):
#     try:
#         tool_dict = tool.dict()

#         # Validate SerperDevTool
#         if tool_dict["name"] == "SerperDevTool" and not (tool_dict.get("api_key") or tool_dict.get("key")):
#             return JSONResponse(
#                 content={"success": False, "message": "API key is required for SerperDevTool."},
#                 status_code=400
#             )

#         # Validate WebsiteSearchTool
#         if tool_dict["name"] == "WebsiteSearchTool" and not tool_dict.get("website_url"):
#             return JSONResponse(
#                 content={"success": False, "message": "URL is required for WebsiteSearchTool."},
#                 status_code=400
#             )

#         tool_dict["created_on"] = datetime.utcnow().isoformat()
#         tool_dict["user_id"] = user_id

#         result = await tools_collection.insert_one(tool_dict)

#         # Include inserted ID in response
#         tool_dict["id"] = str(result.inserted_id)
#         tool_dict.pop("_id", None)

#         return JSONResponse(content={"success": True, "tool": tool_dict}, status_code=201)

#     except Exception as e:
#         return JSONResponse(content={"success": False, "message": f"An error occurred: {e}"}, status_code=500)

# class Tool(BaseModel):
#     id: Optional[str] = Field(default=None, alias="_id")  # MongoDB ObjectId as string
#     name: str
#     api_key: Optional[str] = None
#     website_url: Optional[str] = None
#     type: str
#     tool_name: Optional[str] = None

# @router.post("/tools", status_code=201)
# async def add_tool(tool: Tool, user_id: str = Depends(get_current_user)):
#     try:
#         # Convert input model to dictionary, skipping unset/null values
#         tool_dict = tool.dict(by_alias=True, exclude_unset=True)

#         # Validate SerperDevTool
#         if tool_dict["name"] == "SerperDevTool" and not (tool_dict.get("api_key") or tool_dict.get("key")):
#             return JSONResponse(
#                 content={"success": False, "message": "API key is required for SerperDevTool."},
#                 status_code=400
#             )

#         # Validate WebsiteSearchTool
#         if tool_dict["name"] == "WebsiteSearchTool" and not tool_dict.get("website_url"):
#             return JSONResponse(
#                 content={"success": False, "message": "URL is required for WebsiteSearchTool."},
#                 status_code=400
#             )

#         # Add metadata
#         tool_dict["created_on"] = datetime.utcnow().isoformat()
#         tool_dict["user_id"] = user_id

#         # Search for matching tool in tools_collection1 by name
#         tool1 = await tools_collection1.find_one({"id": tool_dict.get("id")})


#         if tool1 and "name" in tool1:
#             tool_dict["tool_name"] = tool1["name"]  # use the canonical name from collection1
#         else:
#             tool_dict["tool_name"] = None
#         # tool1 = await tools_collection1.find_one({"name": tool_dict["name"]})
#         # if tool1 and "name" in tool1:
#         #     tool_dict["tool_name"] = tool1["name"]  # ✅ Use the name field, not _id
#         # else:
#         #     tool_dict["tool_name"] = None

#         # Insert into tools_collection
#         result = await tools_collection.insert_one(tool_dict)

#         # Convert inserted ObjectId to string for response
#         tool_dict["id"] = str(result.inserted_id)
#         tool_dict.pop("_id", None)  # Just in case _id is in tool_dict

#         return JSONResponse(content={"success": True, "tool": tool_dict}, status_code=201)

#     except Exception as e:
#         return JSONResponse(
#             content={"success": False, "message": f"An error occurred: {e}"},
#             status_code=500
#         )
class Tool(BaseModel):
    id: Optional[str] = None  # <-- remove alias="_id"
    name: str
    api_key: Optional[str] = None
    website_url: Optional[str] = None
    type: str
    tool_name: Optional[str] = None


@router.post("/tools", status_code=201)
async def add_tool(tool: Tool, user_id: str = Depends(get_current_user)):
    try:
        tool_dict = tool.dict(exclude_unset=True)  # keep original field names

        # Validation
        if tool.name == "SerperDevTool" and not (tool.api_key or tool_dict.get("key")):
            return JSONResponse(
                content={"success": False, "message": "API key is required for SerperDevTool."},
                status_code=400
            )

        if tool.name == "WebsiteSearchTool" and not tool.website_url:
            return JSONResponse(
                content={"success": False, "message": "URL is required for WebsiteSearchTool."},
                status_code=400
            )

        tool_dict["created_on"] = datetime.utcnow().isoformat()
        tool_dict["user_id"] = user_id

        # Match tool in tools_collection1 by `id`
        tool1 = await tools_collection1.find_one({"id": tool.id})
        tool_dict["tool_name"] = tool1["name"] if tool1 else None

        result = await tools_collection.insert_one(tool_dict)

        tool_dict["id"] = str(result.inserted_id)
        tool_dict.pop("_id", None)

        return JSONResponse(content={"success": True, "tool": tool_dict}, status_code=201)

    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"An error occurred: {e}"},
            status_code=500
        )