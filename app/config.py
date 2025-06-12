# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Config:
#     DEBUG = os.getenv("DEBUG", "False") == "True"
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     MONGO_URI = os.getenv("MONGO_URI")
#     SERPER_API_KEY = os.getenv("SERPER_API_KEY")
#     AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
#     AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#     BUCKET_NAME = os.getenv("BUCKET_NAME")
#     KMS_KEY_ID = os.getenv("KMS_KEY_ID")
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB file size limit
#     JWT_SECRET = os.getenv("JWT_SECRET")
    
#     # Azure Inference SDK configuration.
#     AZURE_INFERENCE_SDK_KEY = os.getenv("AZURE_INFERENCE_SDK_KEY")
#     AZURE_INFERENCE_SDK_ENDPOINT = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT", "https://zeamedllm8738702350.services.ai.azure.com/models")
#     DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")

#     # DNN Face Detection Files and Settings
#     DNN_MODEL_CONFIG = os.getenv("DNN_MODEL_CONFIG", "face_detector/models/deploy.prototxt")
#     DNN_MODEL_WEIGHTS = os.getenv("DNN_MODEL_WEIGHTS", "face_detector/models/res10_300x300_ssd_iter_140000.caffemodel")
#     FACE_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_DETECTION_CONFIDENCE_THRESHOLD", "0.5"))

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # General Settings
    ENV = os.getenv("ENV", "development").lower()  # "production" or "development"
    DEBUG = os.getenv("DEBUG", "False") == "True"
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB file size limit
    JWT_SECRET = os.getenv("JWT_SECRET")

    # API Keys and URIs
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # AWS Configuration
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    KMS_KEY_ID = os.getenv("KMS_KEY_ID")

    # Azure Inference SDK configuration.
    AZURE_INFERENCE_SDK_KEY = os.getenv("AZURE_INFERENCE_SDK_KEY")
    AZURE_INFERENCE_SDK_ENDPOINT = os.getenv(
        "AZURE_INFERENCE_SDK_ENDPOINT",
        "https://zeamedllm8738702350.services.ai.azure.com/models"
    )
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")

    # DNN Face Detection Files and Settings
    DNN_MODEL_CONFIG = os.getenv("DNN_MODEL_CONFIG", "face_detector/models/deploy.prototxt")
    DNN_MODEL_WEIGHTS = os.getenv("DNN_MODEL_WEIGHTS", "face_detector/models/res10_300x300_ssd_iter_140000.caffemodel")
    FACE_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_DETECTION_CONFIDENCE_THRESHOLD", "0.5"))
