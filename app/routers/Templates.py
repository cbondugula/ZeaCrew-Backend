from fastapi import APIRouter, HTTPException, Depends, Request
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import List, Optional
from pydantic import BaseModel,constr,Field,ValidationError
import os
import logging
from dotenv import load_dotenv
from .llms_fastapi import get_current_user
from crewai import Agent, Task, Crew,Process
from crewai_tools import SerperDevTool,WebsiteSearchTool
from langchain_openai import ChatOpenAI
from datetime import datetime
from crewai import Crew, Process
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from typing import List, Optional
from fastapi.responses import JSONResponse
from crewai import LLM
# , pattern=r'^[a-zA-Z0-9\s]+$'


# Load environment variables
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["agent_db"]
agents_collection = db["agents"]
tasks_collection = db["tasks"]
tools_collection = db["tools"]
llm_connections_collection = db["llm_connections"]
TEMP_collection =db["agents_templates"]
ENRICHED_COLLECTION = db["enriched_agents"]


# Initialize Router
router = APIRouter()

# Schemas
class AgentSchema(BaseModel):
    role: constr( min_length=6, max_length=500 ) # type: ignore
    goal: constr( min_length=6, max_length=500 ) # type: ignore
    backstory: constr( min_length=6, max_length=500 ) # type: ignore
    llms: List[str]
    tools: List[str]
    max_iter: int
    max_rpm: int

class TaskSchema(BaseModel):
    description: constr( min_length=6, max_length=500 ) # type: ignore
    expected_output: constr( min_length=6, max_length=500 ) # type: ignore
    agent_name: constr( min_length=6, max_length=500 ) # type: ignore

class AgentTaskSchema(BaseModel):
    agent: AgentSchema
    task: TaskSchema

class BulkAgentTaskSchema(BaseModel):
    agents: List[AgentTaskSchema]

# class TaskSchema(BaseModel):
#     description: str
#     expected_output: str
#     agent_name: str

class AgentWithTasksSchema(BaseModel):
    agent_id:str
    agent: constr( min_length=6, max_length=500 ) # type: ignore
    role: constr( min_length=6, max_length=500 ) # type: ignore
    goal: constr( min_length=6, max_length=500 ) # type: ignore
    backstory: constr( min_length=6, max_length=500 ) # type: ignore
    llms: List[str]
    tools: List[str]
    max_iter: int
    max_rpm: int
    allow_delegation: bool = False
    tasks: List[TaskSchema]

class MultiAgentDocumentSchema(BaseModel):
    enum:str ### updated today for the for the edit api
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=1000 ) # type: ignore
    llm_provider: constr( min_length=6, max_length=500 ) # type: ignore
    search_provider: constr( min_length=6, max_length=500 ) # type: ignore
    agents: List[AgentWithTasksSchema]

class LLMData(BaseModel):
    id: str
    model: str

class ToolData(BaseModel):
    id: str

class TaskData(BaseModel):
    description: constr( min_length=6, max_length=500 ) # type: ignore
    expected_output: constr( min_length=6, max_length=500 ) # type: ignore
    agent_name: constr( min_length=6, max_length=500 ) # type: ignore

class AgentData(BaseModel):
    role: constr( min_length=6, max_length=500 ) # type: ignore
    goal: constr( min_length=6, max_length=500 ) # type: ignore
    backstory: constr( min_length=6, max_length=500 ) # type: ignore
    max_iter: Optional[int] = 1
    max_rpm: Optional[int] = 10
    llms: Optional[List[LLMData]] = []
    tools: Optional[List[ToolData]] = []

# Define the structure for the template
class Template(BaseModel):
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=500 ) # type: ignore

class AddAndEnrichPayload(BaseModel):
    agent: AgentData
    task: TaskData
    template: Template


@router.post("/chat-agent/initialize-bulk-one-doc")
async def initialize_agents_bulk_one_doc(data: MultiAgentDocumentSchema, user: dict = Depends(get_current_user)):
    try:
        agent_payload = data.dict()
        result = await TEMP_collection.insert_one(agent_payload)

        return {
            "message": "All agents with their tasks stored in a single document",
            "doc_id": str(result.inserted_id),
            "agent_count": len(agent_payload["agents"])
        }
    except Exception as e:
        logging.exception("Multi-agent bulk insert error")
        raise HTTPException(status_code=500, detail=f"Failed to initialize bulk agents: {e}")

from fastapi import Query

@router.get("/chat-agent/all")
async def get_all_chat_agents(
    user: dict = Depends(get_current_user),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    try:
        agents_cursor = TEMP_collection.find({})
        agents_docs = await agents_cursor.to_list(length=None)

        if not agents_docs:
            return {
                "success": False,
                "message": "No multi-agent systems found",
                "systems": []
            }

        total_items = len(agents_docs)
        start = (page - 1) * limit
        end = start + limit
        paginated_docs = agents_docs[start:end]

        result = []

        for doc in paginated_docs:
            system_data = {
                "id": str(doc.get("_id")),
                "title": doc.get("title"),
                "description": doc.get("description"),
                "llm_provider": doc.get("llm_provider"),
                "search_provider": doc.get("search_provider"),
                "agents": []
            }

            for agent in doc.get("agents", []):
                system_data["agents"].append({
                    "role": agent.get("role"),
                    "goal": agent.get("goal"),
                    "backstory": agent.get("backstory"),
                    "llms": agent.get("llms", []),
                    "tools": agent.get("tools", []),
                    "max_iter": agent.get("max_iter", 1),
                    "max_rpm": agent.get("max_rpm", 10),
                    "tasks": [
                        {
                            "description": task.get("description"),
                            "expected_output": task.get("expected_output"),
                            "agent_name": task.get("agent_name")
                        } for task in agent.get("tasks", [])
                    ]
                })

            result.append(system_data)

        return {
            "success": True,
            "page": page,
            "limit": limit,
            "total_items": total_items,
            "total_pages": (total_items + limit - 1) // limit,
            "systems": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching agents: {e}")

   
# @router.get("/chat-agent/all")
# async def get_all_chat_agents(user: dict = Depends(get_current_user)):
#     try:
#         agents_cursor = TEMP_collection.find({})
#         agents_docs = await agents_cursor.to_list(length=None)

#         if not agents_docs:
#             return {
#                 "success": False,
#                 "message": "No multi-agent systems found",
#                 "systems": []
#             }

#         result = []

#         for doc in agents_docs:
#             system_data = {
#                 "id": str(doc.get("_id")),
#                 "title": doc.get("title"),
#                 "description": doc.get("description"),
#                 "llm_provider": doc.get("llm_provider"),
#                 "search_provider": doc.get("search_provider"),
#                 "agents": []
#             }

#             for agent in doc.get("agents", []):
#                 system_data["agents"].append({
#                     "role": agent.get("role"),
#                     "goal": agent.get("goal"),
#                     "backstory": agent.get("backstory"),
#                     "llms": agent.get("llms", []),
#                     "tools": agent.get("tools", []),
#                     "max_iter": agent.get("max_iter", 1),
#                     "max_rpm": agent.get("max_rpm", 10),
#                     "tasks": [
#                         {
#                             "description": task.get("description"),
#                             "expected_output": task.get("expected_output"),
#                             "agent_name": task.get("agent_name")
#                         } for task in agent.get("tasks", [])
#                     ]
#                 })

#             result.append(system_data)

#         return {
#             "success": True,
#             "systems": result
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching agents: {e}")
# Helper: convert to ObjectIds safely

# def to_object_ids(id_list):
#     return [ObjectId(i) for i in id_list if ObjectId.is_valid(i)]

# # Helper: fetch LLMs by ID
# async def get_llm_details(llm_ids, llm_collection):
#     cursor = llm_collection.find({"_id": {"$in": llm_ids}})
#     return [{**llm, "_id": str(llm["_id"])} async for llm in cursor]

# # Helper: fetch Tools by ID
# async def get_tool_details(tool_ids, tool_collection):
#     cursor = tool_collection.find({"_id": {"$in": tool_ids}})
#     return [{**tool, "_id": str(tool["_id"])} async for tool in cursor]

# @router.get("/chat-agent/enriched-all")
# async def get_all_enriched_chat_agents(user: dict = Depends(get_current_user)):
#     try:
#         user_id = user
#         print(f"Fetching systems for user_id: {user_id}")  # DEBUG

#         agents_cursor = ENRICHED_COLLECTION.find({"user_id": user_id})
#         agents_docs = await agents_cursor.to_list(length=None)

#         print(f"Found {len(agents_docs)} systems")  # DEBUG
#         if not agents_docs:
#             return {
#                 "success": True,
#                 "message": "No enriched multi-agent systems found for this user",
#                 "user": user,
#                 "systems": []
#             }

#         result = []

#         for doc in agents_docs:
#             print(f"Processing system ID: {doc.get('_id')}")  # DEBUG
#             system_data = {
#                 "id": str(doc.get("_id")),
#                 "enum": str(doc.get("enum")),
#                 "title": doc.get("title"),
#                 "description": doc.get("description"),
#                 "llm_provider": doc.get("llm_provider"),
#                 "search_provider": doc.get("search_provider"),
#                 "agents": [],
#                 "manager_agent": None
#             }

#             # Process regular agents
#             for agent in doc.get("agents", []):
#                 llm_ids = to_object_ids(agent.get("llms", []))
#                 tool_ids = to_object_ids(agent.get("tools", []))
#                 llms = await get_llm_details(llm_ids, llm_connections_collection)
#                 tools = await get_tool_details(tool_ids, tools_collection)

#                 system_data["agents"].append({
#                     "agent_id": agent.get("agent_id"),
#                     "agent_name": agent.get("agent"),
#                     "role": agent.get("role"),
#                     "goal": agent.get("goal"),
#                     "backstory": agent.get("backstory"),
#                     "llms": llms,
#                     "tools": tools,
#                     "max_iter": agent.get("max_iter", 1),
#                     "max_rpm": agent.get("max_rpm", 10),
#                     "allow_delegation": agent.get("allow_delegation", False),
#                     "tasks": [
#                         {
#                             "description": task.get("description"),
#                             "expected_output": task.get("expected_output"),
#                             "agent_name": task.get("agent_name")
#                         } for task in agent.get("tasks", [])
#                     ]
#                 })

#             # Process manager agent
#             manager = doc.get("manager_agent")
#             if manager:
#                 llm_ids = to_object_ids(manager.get("llms", []))
#                 tool_ids = to_object_ids(manager.get("tools", []))
#                 llms = await get_llm_details(llm_ids, llm_connections_collection)
#                 tools = await get_tool_details(tool_ids, tools_collection)

#                 system_data["manager_agent"] = {
#                     "agent_id": manager.get("agent_id", "manager"),
#                     "agent_name": manager.get("agent"),
#                     "role": manager.get("role"),
#                     "goal": manager.get("goal"),
#                     "backstory": manager.get("backstory"),
#                     "llms": llms,
#                     "tools": tools,
#                     "max_iter": manager.get("max_iter", 1),
#                     "max_rpm": manager.get("max_rpm", 10),
#                     "allow_delegation": manager.get("allow_delegation", True),
#                     "tasks": manager.get("tasks", [])
#                 }

#             result.append(system_data)

#         return {
#             "success": True,
#             "user": user,
#             "systems": result
#         }

#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=f"Error fetching enriched agents: {e}")


@router.get("/chat-agent/enriched-all")
async def get_all_enriched_chat_agents(user: dict = Depends(get_current_user)):
    try:
        user_id = user
        print(f"Fetching systems for user_id: {user_id}")  # DEBUG

        agents_cursor = ENRICHED_COLLECTION.find({"user_id": user_id})
        agents_docs = await agents_cursor.to_list(length=None)

        print(f"Found {len(agents_docs)} systems")  # DEBUG
        if not agents_docs:
            return {
                "success": True,
                "message": "No enriched multi-agent systems found for this user",
                "user": user,
                "systems": []
            }

        result = []

        for doc in agents_docs:
            print(f"Processing system ID: {doc.get('_id')}")  # DEBUG
            print(f"Manager agent raw data: {doc.get('manager_agent')}")  # DEBUG

            # Build system structure
            system_data = {
                "id": str(doc.get("_id")),
                "enum": str(doc.get("enum")),
                "title": doc.get("title"),
                "description": doc.get("description"),
                "llm_provider": doc.get("llm_provider"),
                "search_provider": doc.get("search_provider"),
                "agents": [],
                "manager_agent": None  # Explicit field for manager
            }

            # Add regular agents
            for agent in doc.get("agents", []):
                system_data["agents"].append({
                    "agent_id": agent.get("agent_id"),
                    "agent_name": agent.get("agent"),
                    "role": agent.get("role"),
                    "goal": agent.get("goal"),
                    "backstory": agent.get("backstory"),
                    "llms": agent.get("llms", []),
                    "tools": agent.get("tools", []),
                    "max_iter": agent.get("max_iter", 1),
                    "max_rpm": agent.get("max_rpm", 10),
                    "allow_delegation": agent.get("allow_delegation", False),
                    "tasks": [
                        {
                            "description": task.get("description"),
                            "expected_output": task.get("expected_output"),
                            "agent_name": task.get("agent_name")
                        } for task in agent.get("tasks", [])
                    ]
                })

            # Add manager agent separately
            manager = doc.get("manager_agent")
            if manager:
                print(f"Adding manager agent: {manager.get('agent')}")  # DEBUG
                system_data["manager_agent"] = {
                    "agent_id": manager.get("agent_id", "manager"),
                    "agent_name": manager.get("agent"),
                    "role": manager.get("role"),
                    "goal": manager.get("goal"),
                    "backstory": manager.get("backstory"),
                    "llms": manager.get("llms", []),
                    "tools": manager.get("tools", []),
                    "max_iter": manager.get("max_iter", 1),
                    "max_rpm": manager.get("max_rpm", 10),
                    "allow_delegation": manager.get("allow_delegation", True),
                    "tasks": manager.get("tasks", [])
                }

            result.append(system_data)

        print(f"Returning {len(result)} systems")  # DEBUG
        return {
            "success": True,
            "user": user,
            "systems": result
        }

    except Exception as e:
        print(f"Error: {e}")  # DEBUG
        raise HTTPException(status_code=500, detail=f"Error fetching enriched agents: {e}")



class LLMSchema(BaseModel):
    # name: str
    id: str
    model: str

class ToolSchema(BaseModel):
    id: str


class EnrichmentPayload(BaseModel):
    llms: List[LLMSchema]
    tools: List[ToolSchema]


# Add tool in backend with template 
@router.post("/chat-agent/enrich-from-temp/{template_id}")
async def enrich_agents_from_template(template_id: str, data: EnrichmentPayload, user: str = Depends(get_current_user)):
# async def enrich_agents_from_template(    template_id: str,    data: EnrichmentPayload,    user: dict = Depends(get_current_user)):
    try:
        # Fetch the document using the provided template_id#-enrich-from-temp
        template_doc = await TEMP_collection.find_one({"_id": ObjectId(template_id)})

        if not template_doc:
            return {
                "success": False,
                "message": "Template document not found with the given ID",
                "doc_id": None,
                "agent_count": 0
            }
        agents = template_doc.get("agents", [])
        if not agents:
            return {
                "success": False,
                "message": "No agents found in the template document",
                "doc_id": None,
                "agent_count": 0
            }

        # Add llms and tools to all agents
        for agent in agents:
            agent["llms"] = [llm.dict() for llm in data.llms]
            agent["tools"] = [tool.dict() for tool in data.tools]

        # enriched_payload = {"agents": agents}
        print(template_doc.get("enum"))
        print("-----123test")
        enriched_payload = {
            "enum":template_doc.get("enum"),
            "user_id": user,
            "title": template_doc.get("title"),
            "description": template_doc.get("description"),
            "llm_provider": template_doc.get("llm_provider"),
            "search_provider": template_doc.get("search_provider"),
            "agents": agents
        }

        result = await ENRICHED_COLLECTION.insert_one(enriched_payload)

        # result = await ENRICHED_COLLECTION.insert_one(enriched_payload)

        return {
            "success": True,
            "message": "Agents enriched with LLMs and tools and saved successfully",
            "doc_id": str(result.inserted_id),
            "agent_count": len(agents)
        }

    except Exception as e:
        logging.exception("Failed to enrich agents")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    
from datetime import datetime, timezone
from bson.objectid import ObjectId
from bson.errors import InvalidId
from agent_watch import AgentWatchExtended 
from app.socketHandler import sio 
import re 
# import agentops
# agentops.init(os.getenv("AGENTOPS_API_KEY"))
@router.post("/chat-agent/crew-run/{agent_id}")
# async def run_crew_with_multiple_agents(request: Request, agent_id: str, user_id: str = Depends(get_current_user)):
async def run_crew_with_multiple_agents(request: Request, agent_id: str, user_id: dict = Depends(get_current_user)):
    watch = AgentWatchExtended(model="gpt-4o-mini")
    watch.start()
    start_time = datetime.utcnow()
    try:
        # agentops.init(os.getenv("AGENTOPS_API_KEY"))
        # agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"),tags=[user_id])
        print("Authenticated User ID:", user_id)
        body = await request.json()
        topic = body.get("topic")
        sid = body.get("sid")
        if not sid:
            sid="ABC-Test-123 form Backend code "
        print("Received SID:", sid)

        print(topic)
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required in the request body.")
        
        word_count = len(re.findall(r'\w+', topic))
        if word_count > 2000:
            raise HTTPException(
                status_code=400,
                detail=f"Topic exceeds 2000 word limit. Received {word_count} words.",
            )
        if not sid:
            raise HTTPException(status_code=400, detail="Socket ID (sid) is required")
        await sio.emit("status", {"message": "Verifier started"}, to=sid)
        greeting_patterns = [
            r'^\s*(hi|hello|hey|good morning|good afternoon|good evening|yo|what\'s up|howdy|sup|hey there|hi there|greetings|salutations)\s*$',
            r'^\s*(ok|okay|thanks|thank you|good|bye|good night|goodbye|thankyou|thx|great|ty|cheers|cya|see you|take care|later|peace|farewell)\s*$',
            r'^\s*(great|awesome|cool|nice|fantastic|amazing|wonderful|perfect|good job|well done|fantabulous|brilliant|outstanding)\s*$',
            r'^\s*(yeah|yep|sure|absolutely|of course|totally|sounds good|perfect|you bet|I\'m in|for sure|definitely|okie dokie)\s*$',
            r'^\s*(can you help|could you help|please help|can you assist|need help|help me|assist me|can I get some help)\s*$',
            r'^\s*(how are you|how\'s it going|how are you doing|what\'s new|how have you been|how\'s life|what\'s up with you|how do you do|how’s everything|howdy doody)\s*$',
            r'^\s*(thanks a lot|thanks so much|thanks a million|thank you very much|many thanks|thanks indeed|I appreciate it|I\'m grateful|thanks for that)\s*$',
            r'^\s*(yay|woohoo|yayyy|hurray|awesome sauce|whoo|that\'s amazing|so cool|this is awesome|that\'s great|I\'m so happy)\s*$',
            r'^\s*(yes|yup|yup, absolutely|yes please|right|definitely|sure thing|you bet|exactly|that\'s right|certainly|correct)\s*$',
            r'^\s*(what\'s going on|what\'s up?|how\'s everything going?|how\'s everything with you?|what\'s happening?|what\'s the deal?)\s*$',
            r'^\s*(see ya|take it easy|catch you later|until next time|later alligator|goodbye for now|I\'ll be seeing you|peace out)\s*$',
            r'^\s*(I\'m interested|tell me more|that sounds interesting|I want to know more|I\'m curious|do tell|I\'d love to hear more)\s*$',
            r'^\s*(thanks a bunch|thanks heaps|thanks a ton|thanks for everything|thanks so much for that)\s*$',
        ]


        # Invalid query pattern (numbers/special characters only)
        invalid_query_pattern = r'^\s*[\d@#$%^&*()_+\-=\[\]{}|;:"\\\',.<>?/]+\s*$'

        # Check for greetings
        is_greeting = any(re.match(pattern, topic.lower()) for pattern in greeting_patterns)
        if is_greeting:
            return {
            # "user_id": user_id,
            "success": True,
            "final_report": "You're welcome! How can I assist you with your query today?",
            "duration_seconds": 0,
            "agent_outputs": []
        }

        # Check for invalid queries
        if re.match(invalid_query_pattern, topic):
            return {
                # "user_id": user_id,
                "success": False,
                "final_report": "Please provide a valid query — avoid using only numbers or special characters.",
                "duration_seconds": 0,
                "agent_outputs": []
            }

        try:
            agent_object_id = ObjectId(agent_id)
        except InvalidId:
            agent_object_id = agent_id  # fallback if it’s just a string id


        # Fetch the agent data from the enriched collection
        agent_data = await ENRICHED_COLLECTION.find_one({"_id": agent_object_id})
        logging.info(f"Fetched agent data: {agent_data}")
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent group not found with the provided ID")

        logging.info(f"Fetched agent data: {agent_data}")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Fetch LLM data dynamically from the database
        # agents = agent_data.get("agents", [])
        # if not agents:
        #     raise HTTPException(status_code=404, detail="No agents found in the document")

        # for agent in agents:
        #     llm_data = agent.get("llms")
        #     if not llm_data or len(llm_data) == 0:
        #         raise HTTPException(status_code=404, detail=f"LLM data not found in agent: {agent.get('role')}")
        #     api_keys = {} 
        #     openai_api_key = None
        #     for llm in llm_data:
        #         id = llm["id"]
        #         llm_connection = await llm_connections_collection.find_one({"_id": ObjectId(id)})
        #         # print(llm_connection,"---------------------------------------")--------------------------------

        #         if llm_connection:
        #             env_vars = llm_connection.get("envVariables", {})
        #             for key, value in env_vars.items():
        #                 api_keys[key] = value  # store all keys dynamically
                
        #     # print(f"API Keys for agent {agent.get('role')}: {api_keys}")--------------------------------------------

        #     if not api_keys:
        #         raise HTTPException(status_code=404, detail=f"No API keys found for agent: {agent.get('role')}")

        #     # Example: you can access specific keys like this
        #     openai_key = api_keys.get("OPENAI_API_KEY")    
        #     print(openai_key)
        agents = agent_data.get("agents", [])
        if not agents:
            raise HTTPException(status_code=404, detail="No agents found in the document")
        
        for agent in agents:
            llm_data = agent.get("llms")
            if not llm_data:
                raise HTTPException(status_code=404, detail=f"LLM data not found in agent: {agent.get('role')}")

            llm = None  # final object to be passed
            for llm_entry in llm_data:
                llm_id = llm_entry["id"]

                llm_connection = await llm_connections_collection.find_one({"_id": ObjectId(llm_id)})
                if not llm_connection:
                    continue

                env_vars = llm_connection.get("envVariables", {})
                provider_name = llm_connection.get("provider_name", "").lower()
                model_name = llm_entry.get("model") or (llm_connection.get("Models", [None])[0])

                if not model_name:
                    raise HTTPException(status_code=400, detail=f"Model not specified for LLM connection: {llm_id}")

                # Match provider and assign appropriate LLM
                if provider_name == "openai":
                    openai_api_key = env_vars.get("api_key") or env_vars.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        raise HTTPException(status_code=400, detail="Missing OpenAI API key")
                    llm = ChatOpenAI(model=model_name, api_key=openai_api_key)

                elif provider_name == "gemini":
                    gemini_api_key = env_vars.get("api_key") or env_vars.get("GEMINI_API_KEY")
                    print(f"Gemini API Key: {gemini_api_key}")
                    print("------------------------------------------------------------------GEMINI_API_KEY")
                    if not gemini_api_key:
                        raise HTTPException(status_code=400, detail="Missing Gemini API key")
                    llm = LLM(model=model_name, api_key=gemini_api_key)

                elif provider_name == "anthropic":
                    anthropic_api_key = env_vars.get("api_key") or env_vars.get("ANTHROPIC_API_KEY")
                    if not anthropic_api_key:
                        raise HTTPException(status_code=400, detail="Missing Anthropic API key")
                    llm = LLM(model=model_name, api_key=anthropic_api_key)

                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_name}")

            if not llm:
                raise HTTPException(status_code=404, detail=f"No usable LLM configured for agent: {agent.get('role')}")

            #  Now you can pass this into the agent
            # agent_instance = Agent(..., llm=llm)
            print(f"Agent {agent.get('role')} will use model {model_name} with provider {provider_name}")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Fetch tool data dynamically from the database using the id
        # for agent in agents:
        #     print(" Agent:", agent.get("role"))
        #     tools=[]

        #     tool_data = agent.get("tools")
        #     if not tool_data or len(tool_data) == 0:
        #         raise HTTPException(status_code=404, detail=f"Tool data not found in agent: {agent.get('role')}")

        #     tool_instance = None
        #     for tool in tool_data:
        #         tool_id = tool["id"]  # The id field in the tools collection
        #         print("Tool id:",tool_id,"-----------------------------")
        #         tool_info = await tools_collection.find_one({"_id": ObjectId(tool_id)})  # Lookup by tool_id

        #         print("Tool Info:", tool_info)
                
        #         if tool_info:
        #             serper_api_key = tool_info.get("api_key")  # Extract API key
        #             tool_instance = SerperDevTool(api_key=serper_api_key)
        #             break

        #     if not tool_instance:
        #         raise HTTPException(status_code=404, detail=f"Tool (Serper) API key not found for agent: {agent.get('role')}")
        #     tools.append(tool_instance)
        for agent in agents:
            print("Agent:", agent.get("role"))
            tools = []

            tool_data = agent.get("tools")
            if not tool_data or len(tool_data) == 0:
                raise HTTPException(status_code=404, detail=f"Tool data not found in agent: {agent.get('role')}")

            tool_instance = None
            for tool in tool_data:
                tool_id = tool["id"]
                print("Tool id:", tool_id, "-----------------------------")
                tool_info = await tools_collection.find_one({"_id": ObjectId(tool_id)})
                # tool_info = await tools_collection.find_one({"id": tool_id})
                print("Tool Info:", tool_info)

                if not tool_info:
                    continue  # Skip if tool info not found

                tool_name = tool_info.get("tool_name")
                print("-----------------------------------------------------")
                print(tool_info)
                if tool_name == "SerperDevTool":
                    serper_api_key = tool_info.get("api_key")
                    if serper_api_key:
                        tool_instance = SerperDevTool(api_key=serper_api_key)
                        break  # Assuming one tool is enough
                elif tool_name == "WebsiteSearchTool":
                    website_url = tool_info.get("website_url")
                    if website_url:
                        tool_instance = WebsiteSearchTool(website=website_url)
                        break
            
            if not tool_instance:
                raise HTTPException(status_code=404, detail=f"No valid tool instance found for agent: {agent.get('role')}")

            tools.append(tool_instance)
            print("print the tool name----------------test1123")
            print(tools)
            # agentops.init(trace_name=[tool_name])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        crew_agents = []
        crew_tasks = []
        manager_agent = None
        # Create Prompt Verifier agent
        verifier_agent = Agent(
            role="Prompt Verifier",
            goal="Interpret the user's prompt and select relevant agents",
            backstory="AI expert trained to analyze prompt semantics and map them to Agent roles based on relevance",
            tools=[],
            verbose=True,
            llm=llm
            # llm=ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)
        )

        verifier_task = Task(
        description = (
            f"You are a prompt verifier. The user provided the topic: '{topic}'.\n\n"
            f"Your job is to match this topic with relevant agent roles from the list below.\n"
            f"ONLY return roles that are clearly relevant based on keywords or domain understanding.\n\n"
            f"Roles List:\n{[agent['role'] for agent in agent_data['agents']]}\n\n"
            f" IMPORTANT:\n"
            f"- DO NOT invent or guess new roles.\n"
            f"- ONLY use exact matches from the list above.\n"
            f"- If none are relevant, return an empty list like: []\n\n"
            f" Respond ONLY with a JSON list of matched role strings.\n"
            f"Example: [\"Biomedical Data Scientist\", \"Research Assistant\"]\n\n"
            f"Double-check for relevance. Be strict, but not overly narrow."
        ),
        expected_output="A JSON list of agent roles relevant to the user topic",
        agent=verifier_agent
        )
        verifier_crew = Crew(
            agents=[verifier_agent],
            tasks=[verifier_task],
            verbose=True,
            max_execution_time=3,
            max_iter=1,
            process=Process.sequential
        )

        verifier_result = await verifier_crew.kickoff_async()
        await sio.emit("status", {"message": "Verifier complete"}, to=sid)

        # Extract relevant roles from verifier result
        import json
        try:
            relevant_roles = json.loads(verifier_result.tasks_output[0].raw)
            if not relevant_roles:
                return {"success": False,"message": "The topic does not match any relevant agent roles. Please provide a more relevant prompt."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing relevant roles: {e}\n Relevant agent roles for topic: {verifier_result.tasks_output[0].raw}")
 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        agent_name_mapping = {}
        for agent_info in agent_data["agents"]:
            # await sio.emit("status", {"message": f"Running task for agent {agent_info['agent']}"}, to=sid)
            # await sio.emit("status", {"message": f"List out all agent names {agent_info['agent']}"}, to=sid)
            if agent_info["role"] not in relevant_roles:
                continue
            allow_delegation = str(agent_info.get("allow_delegation", False)).strip().capitalize()  # Capitalize the value
             # Check if the current agent is a manager
            if agent_info["role"].lower() in ["manager", "project manager"]:
                # Extract and set manager LLM and tools
                llm_data = agent_info.get("llms")
                if not llm_data or len(llm_data) == 0:
                    raise HTTPException(status_code=404, detail="LLM data not found for manager agent")

                api_keys = {}
                for llm in llm_data:
                    llm_id = llm["id"]
                    llm_connection = await llm_connections_collection.find_one({"_id": ObjectId(llm_id)})
                    if llm_connection:
                        env_vars = llm_connection.get("envVariables", {})
                        for key, value in env_vars.items():
                            api_keys[key] = value

                # manager_openai_key = api_keys.get("OPENAI_API_KEY")
                # if not manager_openai_key:
                #     raise HTTPException(status_code=404, detail="OpenAI API key not found for manager agent")

                # Define the manager agent
                manager_agent = Agent(
                    role=agent_info.get("role"),
                    goal=agent_info.get("goal"),
                    backstory=agent_info.get("backstory"),
                    allow_delegation=True,
                    verbose=True,
                    llm=llm
                    # llm=ChatOpenAI(model=agent_info["llms"][0]["model"], api_key=manager_openai_key),
                )
                # Skip adding the manager to agents, as it's handled separately
                continue
            await sio.emit("status", {"message": f"Running task for agent {agent_info['agent']}"}, to=sid)
            agent = Agent(
                role=agent_info.get("role"),
                goal=agent_info.get("goal"),
                backstory=agent_info.get("backstory"),
                tools =tools,
                verbose=True,
                allow_delegation=allow_delegation,
                llm=llm
                # llm=ChatOpenAI(model=agent_info["llms"][0]["model"], api_key=openai_api_key),
                  # Use the fetched API key for LLM
            )
            # agent.agent_name = agent_info.get("agent_name", "")
            agent_name_mapping[agent] = agent_info.get("agent", "")
            crew_agents.append(agent)

            for task_info in agent_info.get("tasks", []):
                task_desc = task_info["description"].replace("{user_input}", topic)
                crew_tasks.append(Task(
                    description=task_desc,
                    expected_output=task_info["expected_output"],
                    agent=agent
                ))

        # listener = MyCustomListener()
        crew = Crew(
            agents=crew_agents,
            tasks=crew_tasks,
            verbose=True,
            max_execution_time=15,
            max_iter=3,
            manager_agent=manager_agent if manager_agent else None,
            process=Process.hierarchical if manager_agent else Process.sequential,
            # listeners=[listener]
            )


     
        # Start the execution
        start_time = datetime.now(timezone.utc)
        result = await crew.kickoff_async()
        watch.set_token_usage_from_crew_output(result)
        await sio.emit("final_result", {"message": "Crew execution complete", "result": str(result)}, to=sid)
        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - start_time).total_seconds()
        # watch.end()  # ✅ END THE WATCH
        # summary_text = watch.visualize(method='cli')  # <<< ADD THIS LINE
        # print("=== Agent Watch Summary ===")
        # print(summary_text)

        # def extract_metric(pattern, text, default=None, cast_func=str):
        #     match = re.search(pattern, text)
        #     return cast_func(match.group(1)) if match else default

        # agent_watch_metrics = {
        #     "total_time_seconds": extract_metric(r"Total Time:\s*([\d.]+)", summary_text, 0.0, float),
        #     "input_tokens": extract_metric(r"Input Tokens:\s*(\d+)", summary_text, 0, int),
        #     "output_tokens": extract_metric(r"Output Tokens:\s*(\d+)", summary_text, 0, int),
        #     "total_tokens": extract_metric(r"Total Tokens:\s*(\d+)", summary_text, 0, int),
        #     "cost_usd": extract_metric(r"Cost:\s*\$([\d.]+)", summary_text, 0.0, float),
        #     "average_cpu_usage": extract_metric(r"Average CPU Usage:\s*([\d.]+)%", summary_text, 0.0, float),
        #     "average_memory_mb": extract_metric(r"Average Memory Usage:\s*([\d.]+)", summary_text, 0.0, float),
        #     "carbon_emissions_kg": extract_metric(r"Carbon Emissions:\s*([\d.]+)", summary_text, 0.0, float)
        # }
        watch.end()  # Important to finalize monitoring

        agent_watch_metrics = {
            "total_time_seconds": f"{round(watch.total_time, 2)} sec",
            "input_tokens": f"{watch.input_tokens} tokens",
            "output_tokens": f"{watch.output_tokens} tokens",
            "total_tokens": f"{watch.total_tokens} tokens",
            "cost_usd": f"${round(watch.cost, 6)}",
            "average_cpu_usage": f"{round(sum(watch.cpu_usage) / len(watch.cpu_usage), 2)}%",
            "average_memory_mb": f"{round(sum(watch.memory_usage) / len(watch.memory_usage), 2)} MB",
            "carbon_emissions_kg": f"{round(watch.carbon_emissions, 6)} kg CO₂"
        }

        # print(f"Agent Name: {getattr(crew_tasks[i].agent, 'agent_name', 'N/A')}")
        # Collect outputs
        agent_outputs = []
        if hasattr(result, "tasks_output") and result.tasks_output:
            for i, task_output in enumerate(result.tasks_output):
                agent_outputs.append({
                    "agent_name": agent_name_mapping.get(crew_tasks[i].agent, 'N/A'),
                    # "agent_role": crew_tasks[i].agent.role,
                    # "task_description": crew_tasks[i].description,
                    "expected_output": crew_tasks[i].expected_output,
                    "actual_output": task_output.raw
                })
                print("----------------------------------------------------------------------------------")
                print(f"Agent Role: {crew_tasks[i].agent.role}")
                print(f"Task Description: {crew_tasks[i].description}")
                print(f"Expected Output: {crew_tasks[i].expected_output}")
                print(f"Actual Output: {task_output.raw}")
                print("----------------------------------------------------------------------------------")
        else:
            agent_outputs.append({
                "agent_role": "Unknown",
                "task_description": "Unknown",
                "expected_output": "Unknown",
                "actual_output": str(result)
            })

        # Save the results in the database
        save_data = {
            "user_id": user_id,
            "agent_group_id": agent_id,
            "topic": topic,
            "result": str(result),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "agent_outputs": agent_outputs,
            "agent_watch": agent_watch_metrics,
            "timestamp": datetime.utcnow()
        }
        await db["crew_results"].insert_one(save_data)

        return {
            # "user_id": user_id,
            "success": True,
            "final_report": str(result),
            "duration_seconds": duration_seconds,
            "agent_outputs": agent_outputs,
            "agent_watch": agent_watch_metrics
        }
    
    except Exception as e:
        logging.exception("Crew run error")
        return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"Crew run failed: {e}"
        }
    )
  #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
           
@router.delete("/chat-agent/crew-run/{agent_id}")
async def delete_agent_group(agent_id: str, user: str = Depends(get_current_user)):
    try:
        print(f"User retrieved from token: {user}")
        print(f"Attempting to delete agent group with agent_id: {agent_id}")

        agent_group = await ENRICHED_COLLECTION.find_one({"_id": ObjectId(agent_id)})

        if not agent_group:
            return {
                "success": False,
                "message": "Agent group not found",
                "agent_id": agent_id
            }

        # print("Akhil")
        print(agent_group["user_id"])
        print(user)

        if str(agent_group["user_id"]) != str(user):
            return {
                "success": False,
                "message": "Unauthorized: You can only delete your own agent group",
                "agent_id": agent_id
            }

        result = await ENRICHED_COLLECTION.delete_one({"_id": ObjectId(agent_id)})

        if result.deleted_count == 0:
            return {
                "success": False,
                "message": "Agent group not found or already deleted",
                "agent_id": agent_id
            }

        return {
            "success": True,
            "message": "Agent group deleted successfully",
            "agent_id": agent_id
        }

    except Exception as e:
        logging.exception("Agent group deletion failed")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")
    

# @router.post("/chat-agent/add-and-enrich")
async def add_and_enrich_agent(
    data: AddAndEnrichPayload,
    user: str = Depends(get_current_user)  
):
    try:
        # Prepare agent and task data
        agent_data = data.agent.model_dump()  
        task_data = data.task.model_dump()    

        # Attach task to agent
        agent_data["tasks"] = [task_data]

        # Ensure llms and tools are populated
        agent_data["llms"] = [llm.model_dump() for llm in data.agent.llms]  
        agent_data["tools"] = [tool.model_dump() for tool in data.agent.tools]  

        # Create enriched payload
        enriched_payload = {
            "user_id": user,  # Directly use user ID string
            "title": data.template.title,  # Get title from the request body
            "description": data.template.description , # Set description
            "llm_provider": "OPENAI",  # Set the LLM provider
            "search_provider": "SERPER",  # Set the search provider
            "agents": [agent_data]
        }

        # Insert enriched data directly into ENRICHED_COLLECTION
        enriched_result = await ENRICHED_COLLECTION.insert_one(enriched_payload)

        return {
            "success": True,
            "message": "Agent added, enriched, and stored successfully",
            "enriched_doc_id": str(enriched_result.inserted_id),
            "agent_count": len(enriched_payload.get("agents", []))
        }

    except Exception as e:
        logging.exception("Failed to add and enrich agent")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
# , pattern=r'^[a-zA-Z0-9\s]+$'
class TaskPayload(BaseModel):
    description: constr( min_length=6, max_length=500 ) # type: ignore
    expected_output: constr( min_length=6, max_length=500 ) # type: ignore
    agent_name: constr( min_length=6, max_length=500 ) # type: ignore

class LLMPayload(BaseModel):
    name: Optional[str] = None  # optional if sometimes empty
    config: Optional[dict] = None

class ToolPayload(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None

class AgentPayload(BaseModel):
    role: constr( min_length=6, max_length=500 ) # type: ignore
    goal: constr( min_length=6, max_length=500 ) # type: ignore
    backstory: constr( min_length=6, max_length=500 ) # type: ignore
    llms: List[LLMPayload]
    tools: List[ToolPayload]
    max_iter: int
    max_rpm: int
    tasks: List[TaskPayload]

class AddAndEnrichPayload(BaseModel):
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=500 ) # type: ignore
    llm_provider: str
    search_provider: str
    agents: List[AgentPayload]

class TemplateSchema(BaseModel):
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=500 ) # type: ignore

class AddAndEnrichPayloadMultiAgent(BaseModel):
    agents: List[AgentPayload]  # Reuse your AgentPayload here
    template: TemplateSchema

    
@router.post("/chat-agent/add-and-enrich-multi")
async def add_and_enrich_agent(
    data: AddAndEnrichPayloadMultiAgent,
    user: str = Depends(get_current_user)
):
    try:
        agents_data = []
        for agent in data.agents:
            agent_data = agent.model_dump()
            agent_data["llms"] = [llm.model_dump() for llm in agent.llms]
            agent_data["tools"] = [tool.model_dump() for tool in agent.tools]
            agent_data["tasks"] = [task.model_dump() for task in agent.tasks]
            agents_data.append(agent_data)

        enriched_payload = {
            "user_id": user,
            "title": data.template.title,
            "description": data.template.description,
            "llm_provider": "OPENAI",
            "search_provider": "SERPER",
            "agents": agents_data
        }

        enriched_result = await ENRICHED_COLLECTION.insert_one(enriched_payload)

        return {
            "success": True,
            "message": f"{len(agents_data)} agents added, enriched, and stored successfully",
            "enriched_doc_id": str(enriched_result.inserted_id),
            "agent_count": len(agents_data)
        }

    except Exception as e:
        logging.exception("Failed to add and enrich agents")
        raise HTTPException(status_code=500, detail=f"Error: {e}")



class TaskPayload(BaseModel):
    description: constr( min_length=6, max_length=500 ) # type: ignore
    expected_output: constr( min_length=6, max_length=500 ) # type: ignore
    agent_name: constr( min_length=6, max_length=500 ) # type: ignore

class ToolPayload(BaseModel):
    id: str

class LLMPayload(BaseModel):
    id: str
    model: str
import uuid
from pydantic import BaseModel, Field
class AgentPayload(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent:constr( min_length=6, max_length=500 ) # type: ignore
    role: constr( min_length=6, max_length=500 ) # type: ignore
    goal: constr( min_length=6, max_length=500 ) # type: ignore
    backstory: constr( min_length=6, max_length=500 ) # type: ignore
    llms: List[LLMPayload]
    tools: List[ToolPayload]
    max_iter: int
    max_rpm: int
    tasks: List[TaskPayload]
    allow_delegation: bool = False
from pydantic import BaseModel, constr
class TemplatePayload(BaseModel):
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=500 ) # type: ignore
from typing import Literal
class MultiAgentSystemPayload(BaseModel):
    # enum:str ### updated today for the for the edit api
    # enum: Literal["CUSTOM", "DEFAULT"]
    enum: Literal["CUSTOM"] = "CUSTOM"
    title: constr( min_length=6, max_length=500 ) # type: ignore
    description: constr( min_length=6, max_length=500 )# type: ignore
    llm_provider: str
    search_provider: str
    agents: List[AgentPayload]
    manager_agent: Optional[AgentPayload] = None    




@router.post("/multi-agent-system/create")
async def create_multi_agent_system(
    payload: MultiAgentSystemPayload,
    user: str = Depends(get_current_user)
):
    try:
        # Attach user info to payload
        enriched_payload = payload.dict()
        enriched_payload["user_id"] = user

        # Insert payload into MongoDB collection
        enriched_result = await ENRICHED_COLLECTION.insert_one(enriched_payload)

        return {
            "success": True,
            "message": "Multi-Agent System configuration received successfully.",
            "agent_count": len(payload.agents),
            "system_title": payload.title,
            "user_id": user,
            "inserted_id": str(enriched_result.inserted_id)
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating system: {str(e)}"
        }
    


@router.put("/chat-agent/enriched/{doc_id}")
async def update_enriched_agents(
    doc_id: str,
    data: EnrichmentPayload,
    user: str = Depends(get_current_user)
):
    try:
        existing_doc = await ENRICHED_COLLECTION.find_one({"_id": ObjectId(doc_id)})

        if not existing_doc:
            return {
                "success": False,
                "message": "No enriched document found with the given ID",
                "doc_id": doc_id
            }

        updated_agents = existing_doc.get("agents", [])
        if not updated_agents:
            return {
                "success": False,
                "message": "No agents found to update in this document",
                "doc_id": doc_id
            }

        # Update llms and tools in agents
        for agent in updated_agents:
            agent["llms"] = [llm.dict() for llm in data.llms]
            agent["tools"] = [tool.dict() for tool in data.tools]

        # Perform the update
        update_result = await ENRICHED_COLLECTION.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "agents": updated_agents,
                "updated_by": user
            }}
        )

        if update_result.modified_count == 1:
            return {
                "success": True,
                "message": "Agents' LLMs and tools updated successfully",
                "doc_id": doc_id,
                "agent_count": len(updated_agents)
            }
        else:
            return {
                "success": False,
                "message": "No changes were made to the document",
                "doc_id": doc_id
            }

    except Exception as e:
        logging.exception("Failed to update enriched agents")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# | `enum`    | What Can Be Edited                                         |
# | --------- | ---------------------------------------------------------- |
# | `CUSTOM`  | Full system: title, description, agents, tools, llms, etc. |
# | `DEFAULT` | Only `llms` and `tools` inside each agent                  |



from fastapi.encoders import jsonable_encoder
class LLMModel(BaseModel):
    id: str
    model: Optional[str] = None

class ToolModel(BaseModel):
    id: str
AgentStr = constr( min_length=6, max_length=500 )
class AgentPayload(BaseModel):
    agent_id:Optional[str] = None
    agent: Optional[AgentStr] = None# type: ignore
    role: Optional[AgentStr] = None# type: ignore
    goal: Optional[AgentStr] = None# type: ignore
    backstory: Optional[AgentStr] = None# type: ignore
    llms: List[LLMModel]
    tools: List[ToolModel]
    max_iter: Optional[int] = None
    max_rpm: Optional[int] = None
    allow_delegation: Optional[bool] = False
    tasks: Optional[List[dict]] = None

class MultiAgentSystemPayload(BaseModel):
    enum: str  # "DEFAULT" or "CUSTOM"
    title: Optional[AgentStr] = None# type: ignore
    description: Optional[AgentStr] = None# type: ignore
    llm_provider: Optional[str] = None
    search_provider: Optional[str] = None
    agents: Optional[List[AgentPayload]] = None



@router.put("/multi-agent-system/edit/{system_id}")
async def edit_multi_agent_system(
    system_id: str,
    payload: MultiAgentSystemPayload,
    user: str = Depends(get_current_user)
):
    try:
        object_id = ObjectId(system_id)

        # Fetch existing doc
        # existing = await ENRICHED_COLLECTION.find_one({"_id": object_id, "user_id": user})
        existing = await ENRICHED_COLLECTION.find_one({"_id": object_id})

        if not existing:
            raise HTTPException(status_code=404, detail="System not found or unauthorized.")

        update_data = {"enum": payload.enum, "updated_by": user}

        if payload.enum == "DEFAULT":
            if not payload.agents:
                raise HTTPException(status_code=422, detail="Agents with llms and tools required for DEFAULT update.")

            updated_agents = existing.get("agents", [])
            if not updated_agents:
                return {
                    "success": False,
                    "message": "No agents found to update.",
                    "system_id": system_id
                }

            # Encode llms/tools properly
            llms = [jsonable_encoder(llm) for llm in payload.agents[0].llms]
            tools = [jsonable_encoder(tool) for tool in payload.agents[0].tools]

            for agent in updated_agents:
                agent["llms"] = llms
                agent["tools"] = tools

            update_data["agents"] = updated_agents

        elif payload.enum == "CUSTOM":
            # Validate required fields
            required_fields = ["title", "description", "llm_provider", "search_provider", "agents"]
            for field in required_fields:
                if getattr(payload, field) is None:
                    raise HTTPException(status_code=422, detail=f"Missing required field: {field}")

            # Convert payload safely to dicts
            update_data.update(jsonable_encoder(payload))

        else:
            raise HTTPException(status_code=400, detail="Invalid enum value. Must be 'DEFAULT' or 'CUSTOM'.")

        result = await ENRICHED_COLLECTION.update_one(
            {"_id": object_id},
            {"$set": update_data}
        )

        return {
            "success": True,
            "message": f"Multi-Agent System updated with enum = {payload.enum}.",
            "modified_count": result.modified_count,
            "system_id": system_id
        }

    except Exception as e:
        logging.exception("Edit failed")
        raise HTTPException(status_code=500, detail=f"Error editing system: {str(e)}")


import json

# class AgentPrompt(BaseModel):
#     prompt: str

# def convert_objectid(doc):
#     if isinstance(doc, dict):
#         for key, value in doc.items():
#             if isinstance(value, ObjectId):
#                 doc[key] = str(value)
#             elif isinstance(value, dict):
#                 doc[key] = convert_objectid(value)
#             elif isinstance(value, list):
#                 doc[key] = [convert_objectid(v) for v in value]
#     return doc

# @router.post("/crewai/agent-from-prompt")
# async def generate_agents_from_prompt(data: AgentPrompt, user: dict = Depends(get_current_user)):
#     try:
#         system_prompt = (
#             "You are a CrewAI expert. Based on the given user prompt, generate a structured JSON output with the following fields:\n\n"
#             "Top-level fields:\n"
#             "- enum: always set to 'DEFAULT'\n"
#             "- title: One-line title based on the use case\n"
#             "- description: A concise system description (max 500 chars)\n"
#             "- llm_provider: always 'OPENAI'\n"
#             "- search_provider: always 'SERPER'\n"
#             "- agents: a list of 2-3 agent objects\n\n"
#             "Each agent must include:\n"
#             "- agent_id: use a randomly generated UUID\n"
#             "- agent: agent name\n"
#             "- role\n"
#             "- goal\n"
#             "- backstory\n"
#             "- llms: return empty list\n"
#             "- tools: return empty list\n"
#             "- max_iter: 1\n"
#             "- max_rpm: 10\n"
#             "- allow_delegation: false\n"
#             "- tasks: a list with one task per agent having:\n"
#             "  - description\n"
#             "  - expected_output\n"
#             "  - agent_name (should match agent name)\n\n"
#             "Respond only with a valid JSON object."
#         )

#         llm = ChatOpenAI(model="gpt-4.1-nano", api_key=os.getenv("OPENAI_API_KEY"))
#         response = llm.invoke(f"{system_prompt}\n\nUser Prompt: {data.prompt}")

#         try:
#             response_data = json.loads(response.content)
#         except Exception as parse_err:
#             raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {parse_err}")

#         # Insert into MongoDB
#         insert_result = await TEMP_collection.insert_one(response_data)

#         return {
#             "message": "Agents successfully generated and stored",
#             "doc_id": str(insert_result.inserted_id),
#             "agent_count": len(response_data.get("agents", [])),
#             "data": convert_objectid(response_data)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agent creation failed: {e}")
# from openai import OpenAI 
# class Message(BaseModel):
#     role: str
#     content: str

# class AgentPrompt(BaseModel):
#     # Now, 'messages' is mandatory and is always a list of Message objects
#     messages: List[Message] = Field(..., min_length=1, description="A list of conversation messages, at least one message is required.")

# # --- Existing convert_objectid function ---
# def convert_objectid(doc):
#     if isinstance(doc, dict):
#         for key, value in doc.items():
#             if isinstance(value, ObjectId):
#                 doc[key] = str(value)
#             elif isinstance(value, dict):
#                 doc[key] = convert_objectid(value)
#             elif isinstance(value, list):
#                 doc[key] = [convert_objectid(v) for v in value]
#     return doc

# @router.post("/crewai/agent-from-prompt")
# async def generate_agents_from_prompt(data: AgentPrompt, user: dict = Depends(get_current_user)):
#     try:
#         system_prompt_content = (
#             "You are a CrewAI expert. Based on the given user prompt, generate a structured JSON output with the following fields:\n\n"
#             "Top-level fields:\n"
#             "- enum: always set to 'DEFAULT'\n"
#             "- title: One-line title based on the use case\n"
#             "- description: A concise system description (max 500 chars)\n"
#             "- llm_provider: always 'OPENAI'\n"
#             "- search_provider: always 'SERPER'\n"
#             "- agents: a list of 2-3 agent objects\n\n"
#             "Each agent must include:\n"
#             "- agent_id: use a randomly generated UUID\n"
#             "- agent: agent name\n"
#             "- role\n"
#             "- goal\n"
#             "- backstory\n"
#             "- llms: return empty list\n"
#             "- tools: return empty list\n"
#             "- max_iter: 1\n"
#             "- max_rpm: 10\n"
#             "- allow_delegation: false\n"
#             "- tasks: a list with one task per agent having:\n"
#             " - description\n"
#             " - expected_output\n"
#             " - agent_name (should match agent name)\n\n"
#             "Respond only with a valid JSON object."
#         )

#         conversation_history = [{"role": "system", "content": system_prompt_content}]

#         if data.prompt:
#             # If a single prompt is provided, add it as a user message
#             conversation_history.append({"role": "user", "content": system_prompt_content})
#         elif data.messages:
#             # If conversational messages are provided, extend the history
#             conversation_history.extend([msg.model_dump() for msg in data.messages])
#         else:
#             # This case should ideally be caught by the Pydantic validator,
#             # but it's good for robust error handling.
#             raise HTTPException(status_code=400, detail="No prompt or messages provided.")

#         # Initialize OpenAI client
#         # It's better to initialize this once globally if possible, or use a dependency injection.
#         client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         completion = client.chat.completions.create(
#             model="gpt-4o-mini", # Using gpt-4o-mini as a robust general-purpose model
#             messages=conversation_history,
#             response_format={"type": "json_object"}
#         )
#         response_content = completion.choices[0].message.content

#         try:
#             response_data = json.loads(response_content)
#         except json.JSONDecodeError as parse_err:
#             raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {parse_err}. Content: {response_content}")

#         # Insert into MongoDB
#         insert_result = await TEMP_collection.insert_one(response_data)

#         return {
#             "message": "Agents successfully generated and stored",
#             "doc_id": str(insert_result.inserted_id),
#             "agent_count": len(response_data.get("agents", [])),
#             "data": convert_objectid(response_data)
#         }

#     except ValidationError as ve:
#         raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
#     except Exception as e:
#         print(f"Error during agent creation: {e}") # Log the error for debugging
#         raise HTTPException(status_code=500, detail=f"Agent creation failed: {e}")

from fastapi import APIRouter, Depends, HTTPException, status # 'status' import added for clarity
from typing import List, Dict, Any, Optional
from openai import OpenAI # Correct import for the OpenAI Python client
class Message(BaseModel):
    """Represents a single message in a conversation."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'system').")
    content: str = Field(..., description="Content of the message.")

class AgentPrompt(BaseModel):
    """
    Input model for generating CrewAI agents, expecting a list of conversational messages.
    This model now strictly uses the 'messages' list for all inputs (single or multi-turn).
    """
    messages: List[Message] = Field(..., min_length=1, description="A list of conversation messages. At least one message is required.")

# --- Utility Function for MongoDB ObjectId Conversion ---
def convert_objectid(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts MongoDB ObjectId instances within a dictionary (and its nested
    dicts/lists) to their string representation, for JSON serialization.
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, dict):
                doc[key] = convert_objectid(value)
            elif isinstance(value, list):
                doc[key] = [convert_objectid(v) for v in value]
    return doc

@router.post("/crewai/agent-from-prompt", summary="Generate CrewAI Agents from Prompt")
async def generate_agents_from_prompt(data: AgentPrompt, user: dict = Depends(get_current_user)):
    """
    Takes a user prompt (or conversation history) and, using an LLM, generates
    a structured JSON output defining CrewAI agents and their tasks.
    This generated data is then stored in MongoDB.
    """
    # Check if MongoDB connection was successful at startup
    if TEMP_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not established. Please check server logs."
        )

    try:
        # Define the system prompt that guides the LLM to produce the desired JSON structure.
        # This prompt is crucial for instructing the AI on the expected output format.
        system_prompt_content = (
            "You are a CrewAI expert. Based on the given user prompt, generate a structured JSON output with the following fields:\n\n"
            "Top-level fields:\n"
            "- enum: always set to 'DEFAULT'\n"
            "- title: One-line title based on the use case\n"
            "- description: A concise system description (max 500 chars)\n"
            "- llm_provider: always 'OPENAI'\n"
            "- search_provider: always 'SERPER'\n"
            "- agents: a list of 2-3 agent objects\n\n"
            "Each agent must include:\n"
            "- agent_id: use a randomly generated UUID (Python's uuid.uuid4() equivalent). This should be a string.\n"
            "- agent: agent name (e.g., 'Market Analyst')\n"
            "- role: agent's role (e.g., 'Senior Market Research Analyst')\n"
            "- goal: agent's goal (e.g., 'Identify key market trends and opportunities')\n"
            "- backstory: agent's backstory (e.g., 'Has 10 years of experience in tech market analysis')\n"
            "- llms: return empty list\n"
            "- tools: return empty list\n"
            "- max_iter: 1\n"
            "- max_rpm: 10\n"
            "- allow_delegation: false\n"
            "- tasks: a list with one task per agent having:\n"
            "  - description: detailed description of the task\n"
            "  - expected_output: what the task is expected to produce\n"
            "  - agent_name: (should match the 'agent' field above)\n\n"
            "Respond ONLY with a valid JSON object. Ensure agent_id is a valid UUID string."
        )

        # Construct the full conversation history for the LLM.
        # This includes the guiding system prompt followed by the user's input messages.
        conversation_history = [{"role": "system", "content": system_prompt_content}] + \
                               [msg.model_dump() for msg in data.messages]

        # Get OpenAI API key from environment variables. Essential for authentication.
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPENAI_API_KEY environment variable not set. Please configure it."
            )

        # CORRECTED: Initialize the direct OpenAI client.
        # This is the client that has the .chat.completions.create() method.
        client_openai = OpenAI(api_key=openai_api_key)

        # Make the API call to OpenAI's chat completion endpoint.
        completion = client_openai.chat.completions.create(
            model="gpt-4o-mini", # A good balance of cost and capability for this task
            messages=conversation_history,
            response_format={"type": "json_object"} # Explicitly request JSON output
        )
        response_content = completion.choices[0].message.content # Extract content from the LLM's response

        try:
            # Attempt to parse the LLM's string response into a JSON object.
            response_data = json.loads(response_content)
        except json.JSONDecodeError as parse_err:
            # Handle cases where the LLM does not return valid JSON, which can happen.
            print(f"LLM returned invalid JSON: {parse_err}. Content received: '{response_content}'")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM returned invalid JSON. Error: {parse_err}. LLM output: {response_content[:200]}..." # Show snippet
            )

        # Insert the successfully parsed structured data into MongoDB.
        # This stores the generated CrewAI configuration.
        insert_result = await TEMP_collection.insert_one(response_data)

        # Return a success response, including key details and the generated data.
        return {
            "success": True,
            "message": "Agents successfully generated and stored",
            "doc_id": str(insert_result.inserted_id), # Convert ObjectId to string for JSON serialization
            "agent_count": len(response_data.get("agents", [])), # Count agents for convenience
            "data": convert_objectid(response_data) # Ensure all ObjectIds are strings in the returned data
        }

    except ValidationError as ve:
        # Catch Pydantic validation errors (e.g., empty messages list, malformed message)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "message": "Invalid input data format",
                "errors": ve.errors()
            }
        )

    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred during agent creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "Agent creation failed due to an internal server error",
                "error": str(e)
            }
        )

# --- FastAPI Endpoint (Updated System Prompt) ---
@router.post("/crewai/agent-from-prompt_m", summary="Generate CrewAI Agents from Prompt")
async def generate_agents_from_prompt(data: AgentPrompt, user: dict = Depends(get_current_user)):
    if TEMP_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not established. Please check server logs."
        )

    try:
        # --- MODIFIED SYSTEM PROMPT CONTENT ---
        system_prompt_content = (
            "You are a CrewAI expert. Based on the given user prompt, generate a structured JSON output with the following fields:\n\n"
            "Top-level fields:\n"
            "- enum: always set to 'DEFAULT'\n"
            "- title: One-line title based on the use case\n"
            "- description: A concise system description (max 500 chars)\n"
            "- llm_provider: always 'OPENAI'\n"
            "- search_provider: always 'SERPER'\n"
            "- agents: a list of 2-3 agent objects (excluding the manager)\n" # Clarified for LLM
            "- manager_agent: a single agent object for the crew manager\n\n" # NEW FIELD
            "Each agent (including the manager) must include:\n"
            "- agent_id: use a randomly generated UUID (Python's uuid.uuid4() equivalent). This should be a string.\n"
            "- agent: agent name (e.g., 'Market Analyst', 'Project Lead')\n"
            "- role: agent's role (e.g., 'Senior Market Research Analyst', 'Crew Manager')\n"
            "- goal: agent's goal (e.g., 'Identify key market trends and opportunities', 'Oversee crew operations')\n"
            "- backstory: agent's backstory (e.g., 'Has 10 years of experience in tech market analysis', 'Experienced project leader')\n"
            "- llms: return empty list\n"
            "- tools: return empty list\n"
            "- max_iter: 1\n"
            "- max_rpm: 10\n"
            "- allow_delegation: (for regular agents: false, for manager_agent: true)\n" # Clarified for LLM
            "- tasks: (Only for regular agents, not for the manager_agent): a list with one task per agent having:\n"
            "  - description: detailed description of the task\n"
            "  - expected_output: what the task is expected to produce\n"
            "  - agent_name: (should match the 'agent' field above)\n\n"
            "Important Notes for Manager Agent:\n"
            "- The `manager_agent` should have `allow_delegation: true`.\n"
            "- The `manager_agent` should NOT have a `tasks` field.\n"
            "- The `manager_agent`'s role and goal should reflect management responsibilities.\n\n"
            "Respond ONLY with a valid JSON object. Ensure agent_id is a valid UUID string."
        )
        # --- END MODIFIED SYSTEM PROMPT ---

        conversation_history = [{"role": "system", "content": system_prompt_content}] + \
                               [msg.model_dump() for msg in data.messages]

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPENAI_API_KEY environment variable not set. Please configure it."
            )

        client_openai = OpenAI(api_key=openai_api_key)

        completion = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content

        try:
            response_data = json.loads(response_content)
        except json.JSONDecodeError as parse_err:
            print(f"LLM returned invalid JSON: {parse_err}. Content received: '{response_content}'")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM returned invalid JSON. Error: {parse_err}. LLM output: {response_content[:200]}..."
            )

        insert_result = await TEMP_collection.insert_one(response_data)

        return {
            "success": True,
            "message": "Agents successfully generated and stored",
            "doc_id": str(insert_result.inserted_id),
            "agent_count": len(response_data.get("agents", [])),
            "data": convert_objectid(response_data)
        }

    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "message": "Invalid input data format",
                "errors": ve.errors()
            }
        )

    except Exception as e:
        print(f"An unexpected error occurred during agent creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "Agent creation failed due to an internal server error",
                "error": str(e)
            }
        )
    

DRAFT_STORE = {}

# Models
class LLMPayload(BaseModel):
    id: str
    model: str

class ToolPayload(BaseModel):
    id: str

class TaskPayload(BaseModel):
    description: constr(min_length=6, max_length=500)# type: ignore
    expected_output: constr(min_length=6, max_length=500)# type: ignore
    agent_name: constr(min_length=6, max_length=500)# type: ignore

class AgentPayload(BaseModel):
    agent_id: str = uuid.uuid4().hex
    agent: constr(min_length=6, max_length=500)# type: ignore
    role: constr(min_length=6, max_length=500)# type: ignore
    goal: constr(min_length=6, max_length=500)# type: ignore
    backstory: constr(min_length=6, max_length=500)# type: ignore
    llms: List[LLMPayload]
    tools: List[ToolPayload]
    max_iter: int = 1
    max_rpm: int = 1
    tasks: List[TaskPayload] = []
    allow_delegation: bool = False

class MultiAgentSystemPayload(BaseModel):
    enum: Literal["CUSTOM"] = "CUSTOM"
    title: constr(min_length=6, max_length=500)# type: ignore
    description: constr(min_length=6, max_length=500)# type: ignore
    process_type: Literal["Sequential", "Hierarchical"]
    llm_provider: str
    search_provider: str
    agents: List[AgentPayload]
    manager_agent: Optional[AgentPayload] = None

# Draft create
@router.post("/draft/create")
def create_draft():
    draft_id = uuid.uuid4().hex
    DRAFT_STORE[draft_id] = {}
    return {"draft_id": draft_id}

# Draft update
@router.put("/draft/{draft_id}")
def update_draft(draft_id: str, payload: dict):
    if draft_id not in DRAFT_STORE:
        raise HTTPException(status_code=404, detail="Draft not found")
    DRAFT_STORE[draft_id] = payload
    return {"success": True}

# Draft get
@router.get("/draft/{draft_id}")
def get_draft(draft_id: str):
    if draft_id not in DRAFT_STORE:
        raise HTTPException(status_code=404, detail="Draft not found")
    return DRAFT_STORE[draft_id]

# Final submit
@router.post("/multi-agent-system/submit/{draft_id}")
def submit_system(draft_id: str):
    if draft_id not in DRAFT_STORE:
        raise HTTPException(status_code=404, detail="Draft not found")
    data = DRAFT_STORE[draft_id]
    try:
        system_payload = MultiAgentSystemPayload(**data)
        del DRAFT_STORE[draft_id]
        return {"success": True, "message": "System deployed", "data": system_payload.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))