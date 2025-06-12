import socketio

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

@sio.event
async def connect(sid, environ, auth):
    print(f"Client connected: {sid}")
    token = auth.get('token') if auth else None
    if not token:
        raise ConnectionRefusedError('authentication failed')
    else:
        await sio.emit("message", f"Welcome client {sid}!", to=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def chat_message(sid, data):
    print(f"Message from {sid}: {data}")
    await sio.emit("chat_message", data)

 