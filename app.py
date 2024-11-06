from bpdata.main import app
import os
from dotenv import load_dotenv
load_dotenv()


HOST = os.getenv('HOST', '0.0.0.0')
PORT = os.getenv('PORT', 8000)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
