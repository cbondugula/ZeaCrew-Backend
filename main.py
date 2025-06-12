from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from app.socketHandler import sio
import socketio
import uvicorn
import logging

# Basic logging configuration (add this if not already present)
logging.basicConfig(level=logging.INFO)

from app.config import Config
from app.routers import llms_fastapi,integrations_fastapi,tools_fastapi,Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Secure Headers Middleware_a
class SecureHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

# SlowAPI Rate Limiting
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])

app = FastAPI()

app.state.limiter = limiter

app.add_middleware(SecureHeadersMiddleware)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://zealab.ai", "http://zeamed-agent.s3-website-us-east-1.amazonaws.com"],
    # allow_origins=["https://zealab.ai", "http://zeamed-doctor-testing.s3-website-us-east-1.amazonaws.com","http://localhost:4200","http://zeamed-agent.s3-website-us-east-1.amazonaws.com"],
    allow_origins=[
        "https://zealab.ai",
        "http://zeamed-doctor-testing.s3-website-us-east-1.amazonaws.com",
        "http://zeamed-agent.s3-website-us-east-1.amazonaws.com",
        "http://localhost:4200",
        "http://localhost:8100",
        "capacitor://localhost",
        "ionic://localhost",
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:5500",
        "https://localhost:8100",
        "http://localhost:3000",
        "https://localhost"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# socket_app = socketio.ASGIApp(sio, app, socketio_path="/events/")

# socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
socket_app = socketio.ASGIApp(sio, app, socketio_path="/events/")

# socket_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/events/")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Completed with status: {response.status_code}")
    return response

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"success": False, "detail": "Rate limit exceeded. Please try again later."}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Internal server error."}
    )
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"success": False, "detail": exc.errors()}
    )

# from my_listener import ws_router, MyCustomListener
# from app.routers.Templates import Crew  # or wherever your crew is defined
# from fastapi.staticfiles import StaticFiles

# app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(llms_fastapi.router, prefix="/llms")
app.include_router(integrations_fastapi.router, prefix="/itgs")
app.include_router(tools_fastapi.router, prefix="/tls")
app.include_router(Templates.router, prefix="/temp")
# app.include_router(ws_router)  # WebSocket route for real-time updates
# listener = MyCustomListener()
# Crew.setup_listener(listener)


# @app.on_event("startup")
# async def startup_event():
#     try:
#         await create_indexes()
#     except Exception as e:
#         logging.error(f"Startup failed: {e}")
#         raise

@app.get("/")
async def root():
    return {"success": True, "message": "Hello, FastAPI! Your app is running."}

if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
    uvicorn.run("main:socket_app", host="0.0.0.0", port=5000, reload=True)
