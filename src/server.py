import aiohttp
import asyncio
import uvicorn
import cv2
import pathlib
import os
from io import BytesIO
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from detect import *

weights_file_url = 'https://github.com/Kata0Vankee/trained_models/blob/master/yolov3.weights?raw=true'
weights_file_name = 'yolov3.weights'

path = pathlib.Path(__file__).parent

app = Starlette()
app.mount('/static', StaticFiles(directory='src/static'))
app.mount('/img', StaticFiles(directory='src/img'))

async def download_file(url, dest):
	if dest.exists(): return
	async with aiohttp.ClientSession() as session:
		async with session.get(url) as response:
			data = await response.read()
			with open(dest, 'wb') as f:
				f.write(data)


def save_image(path, img_bytes, image_name):
    name = image_name + '.jpg'
    img = Image.open(BytesIO(img_bytes))
    if (path/'img'/name).exists():
        os.remove(path/'img'/name)
        os.remove(path/'img'/'pic1_detected.jpg')
    img.save(path/'img'/name)


#Load the model
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(download_file(weights_file_url, path / weights_file_name))]
done = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def detect(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    save_image(path, img_bytes, 'pic1')
    detect_img()
    return JSONResponse({'result': 'img/pic1_detected.jpg'})


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")