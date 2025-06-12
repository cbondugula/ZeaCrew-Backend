# from fastapi import Depends, HTTPException
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import jwt
# from app.config import Config
# import logging

# # HTTPBearer automatically extracts the token without needing to split it.
# security = HTTPBearer()

# def decode_jwt(token: str):
#     try:
#         # Directly decode the token received from the dependency.
#         return jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
#     except Exception as e:
#         logging.exception("JWT decoding failed: %s", str(e))
#         raise HTTPException(status_code=401, detail="Invalid token")

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     token = credentials.credentials  # This is the complete token.
#     payload = decode_jwt(token)
#     if not payload.get("id"):
#         raise HTTPException(status_code=401, detail="Invalid token: missing id")
#     return payload
from fastapi import Header, HTTPException, Depends
import jwt
from app.config import Config
import logging

async def get_current_user(authorization: str = Header(...)):
    """
    Dependency to extract and decode the JWT token provided directly in the
    Authorization header. The header value should be only the token string.
    """
    token = authorization.strip()  # Expecting the token directly, without "Bearer "
    try:
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
    except Exception as e:
        logging.exception("JWT decoding failed: %s", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if "id" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token: missing id")
    
    return payload
