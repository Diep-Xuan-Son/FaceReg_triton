import os, sys
from pathlib import Path
import json 
import numpy as np
import cv2
from io import BytesIO
import shutil
import threading
import uvicorn
import redis
import time
from datetime import datetime
from string import ascii_letters, digits

from fastapi import FastAPI, Request, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from schemes import *
from triton_services import *
from libs.utils import *

from service_ai.spoof_detection import SpoofDetectionRunnable
from service_ai.spoof_detection_onnx import FakeFace

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_AVATAR = "static/avatar"
PATH_IMG_AVATAR = f"{str(ROOT)}/{IMG_AVATAR}"
PATH_IMG_FAIL = f"{str(ROOT)}/static/failface"
check_folder_exist(path_avatar=PATH_IMG_AVATAR, path_imgfail=PATH_IMG_FAIL)

heart_beat_thread = threading.Thread(target=delete_file_cronj, args=(PATH_IMG_FAIL, 25200), daemon=True)
heart_beat_thread.start()

# SPOOFINGDET = SpoofDetectionRunnable(**{"model_path": f"{str(ROOT)}/weights/spoofing.pt",
# 									"imgsz": 448,
# 									"device": 'cpu',
# 									"cls_names": ['authentic', 'fake']})
SPOOFINGDET = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")

TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
REDISSERVER_IP = os.getenv('REDISSERVER_IP', "192.168.6.163")
REDISSERVER_PORT = os.getenv('REDISSERVER_PORT', 6400)
print("----TRITONSERVER_IP: ", TRITONSERVER_IP)
print("----TRITONSERVER_PORT: ", TRITONSERVER_PORT)
print("----REDISSERVER_IP: ", REDISSERVER_IP)
print("----REDISSERVER_PORT: ", REDISSERVER_PORT)
tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
redisClient = redis.StrictRedis(host=REDISSERVER_IP,
								port=int(REDISSERVER_PORT),
								password="RedisAuth",
								db=0)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/registerFace")
async def registerFace(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	try:
		code = params.code
		print(code)
		special_letters = set(code).difference(ascii_letters + digits)
		if special_letters:
			return {"success": False, "error_code": 8010, "error": "There are some special letters in user code!"}
		if redisClient.hexists("FaceInfor1", code):
			return {"success": False, "error_code": 8004, "error": "This user has been registered!"}

		path_avatar = f"{IMG_AVATAR}/{code}/face_1.jpg"
		path_code = os.path.join(PATH_IMG_AVATAR, code)
		if os.path.exists(path_code):
			shutil.rmtree(path_code)
		os.mkdir(path_code)

		name = params.name
		birthday = params.birthday
		imgs = []
		img_infor = []

		for i, image in enumerate(images):
			image_byte = await image.read()
			nparr = np.fromstring(image_byte, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			img_infor.append(img.shape[:2])
			cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
			# img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
			imgs.append(img)
		imgs = np.array(imgs)
		img_infor = np.array(img_infor)
		#---------------------------face det-------------------------
		in_retinaface, out_retinaface = get_io_retinaface(imgs)
		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
		croped_image = results.as_numpy("croped_image")
		num_object = results.as_numpy("num_obj").squeeze(1)
		print("-----num_object: ", num_object)
		many_face_index = (np.where(num_object>1)[0]+1).tolist()
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		if len(many_face_index)!=0:
			return {"success": False, "error_code": 8007, "error": f"Too many faces in image number {str(many_face_index).strip('[]')}"}
		# print(croped_image.shape)
		# cv2.imwrite("sadas.jpg", croped_image[0])
		#////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		in_arcface, out_arcface = get_io_ghostface(croped_image)
		results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_arcface, outputs=out_arcface)
		feature = results.as_numpy("feature_norm")
		feature = feature.astype(np.float16)
		print("------arcface_feature: ", feature.shape)
		
		# in_ghostface, out_ghostface = get_io_ghostface(croped_image)
		# results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
		# feature = results.as_numpy("feature_norm")
		# feature = feature.astype(np.float16)
		# print(feature.shape)

		# dt_person_information = {code: f"{code},./{name},./{birthday}"}
		# dt_person_feature = {code: feature.tobytes()}
		redisClient.hset("FaceInfor1", code, f"{code}@@@{name}@@@{birthday}@@@{path_avatar}")
		redisClient.hset("FaceFeature1", code, feature.tobytes())

		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteUser")
def deleteUser(codes: List[str] = ["001099008839"]):
	print(codes)
	try:
		codes_noregister = []
		for code in codes:
			if not redisClient.hexists("FaceInfor1", code) or not redisClient.hexists("FaceFeature1", code):
				codes_noregister.append(code)
		if len(codes_noregister)>0:
			return {"success": False, "error_code": 8006, "error": f"User {tuple(codes_noregister)} has not been registered!"}

		redisClient.hdel("FaceInfor1", *codes)
		redisClient.hdel("FaceFeature1", *codes)

		for code in codes:
			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteAllUser")
def deleteAllUser():
	try:
		redisClient.delete("FaceInfor1")
		redisClient.delete("FaceFeature1")
		if os.path.exists(PATH_IMG_AVATAR):
			shutil.rmtree(PATH_IMG_AVATAR)
			os.mkdir(PATH_IMG_AVATAR)
		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/getInformationUser")
def getInformationUser(codes: List[str] = []):
	try:
		infor_persons = {}
		print(codes)
		if len(codes)==0:
			key_infor_persons = redisClient.hkeys("FaceInfor1")
			if len(key_infor_persons)==0:
				return {"success": True, "information": infor_persons}
			key_infor_persons = b'-;'.join(key_infor_persons).decode('utf-8').split("-;")
			val_infor_persons = redisClient.hvals("FaceInfor1")
			val_infor_persons = np.array(b'@@@'.join(val_infor_persons).decode('utf-8').split("@@@")).reshape(-1,4)	# shape (-1,3) for 3 field: code, name, birthday
			infor_persons = dict(zip(key_infor_persons, val_infor_persons.tolist()))
		else:
			for code in codes:
				print(redisClient.hexists("FaceInfor1", code))
				if not redisClient.hexists("FaceInfor1", code):
					infor_persons[code] = "No register"
					continue
				infor_person = redisClient.hget("FaceInfor1", code)
				infor_person = infor_person.decode("utf-8").split("@@@")
				infor_persons[code] = {"id": infor_person[0], \
										"name": infor_person[1], \
										"birthday": infor_person[2], \
										"avatar": infor_person[3]
										}
		return {"success": True, "information": infor_persons}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/searchUser")
async def searchUser(image: UploadFile = File(...)):
	id_faces = redisClient.hkeys("FaceFeature1")
	if len(id_faces) == 0:
		return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}
	image_byte = await image.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	#----------------
	# img_list = os.listdir("./image_error")
	# cv2.imwrite(f"aaa.jpg", img)
	#------------------
	t_det = time.time()
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(img)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	croped_image = results.as_numpy("croped_image")
	num_object = results.as_numpy("num_obj").squeeze(1)
	print("-----num_object: ", num_object)
	if len(croped_image)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		
	box = results.as_numpy("box")[0]
	# print((box[2]-box[0])*(box[3]-box[1]))
	area_img = img.shape[0]*img.shape[1]
	w_crop = (box[2]-box[0])
	h_crop = (box[3]-box[1])
	# if not area_img*0.1<w_crop*h_crop<area_img*0.3:
	# 	return {"success": False, "error_code": 8001, "error": "Face size is not true"}
	#---------------spoofing--------------
	box_expand = np.array([max(box[0]-w_crop,0), max(box[1]-h_crop,0), min(box[2]+w_crop, img.shape[1]), min(box[3]+h_crop, img.shape[0])], dtype=int)
	result = SPOOFINGDET.inference([img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
	print("---------result_spoofing", result)
	# cv2.imwrite(f"aaa.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
	if result[1] > 0.78:
		#img_list = os.listdir("./image_spoofing")
		#cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
		return {"success": False, "error_code": 8002, "error": "Fake face image"}
	#//////////////////////////////////////
	#////////////////////////////////////////////////////////////
	print("------Duration det: ", time.time()-t_det)

	t_reg = time.time()
	#---------------------------face reg-------------------------
	in_arcface, out_arcface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_arcface, outputs=out_arcface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	print("------arcface_feature: ", feature.shape)

	# in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	# results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	# feature = results.as_numpy("feature_norm")
	# feature = feature.astype(np.float16)
	#////////////////////////////////////////////////////////////
	print("------Duration reg: ", time.time()-t_reg)

	#---------------------------compare fail face---------------------
	ft_faces = np.array(redisClient.hvals("FaceFeatureFail2"))
	if len(ft_faces)==0:
		codes_fail = []
	else:
		feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)
		# print(feature_truth.shape)
		# print(feature.shape)
		dist = np.linalg.norm(feature - feature_truth, axis=1)
		similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

		ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeatureFail2"), encoding="utf-8").astype(int)
		code_list = np.char.decode(redisClient.hvals("FaceListCodeFail2"), encoding="utf-8")

		codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
		ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
		similarity = similarity[ft_faces_idx_sort]				# value with sorted key
		similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
		# print(similarity_average)
		rand = np.random.random(similarity_average.size)
		idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
		# print(idx_sorted)
		similaritys = similarity_average[idx_sorted]
		codes_fail = codes[idx_sorted][similaritys>0.8]
	#/////////////////////////////////////////////////////////////////

	t_db = time.time()
	ft_faces = np.array(redisClient.hvals("FaceFeature1"))
	# print(ft_faces)
	# print(len(ft_faces))
	feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), -1, 512)
	print("------Duration db: ", time.time()-t_db)

	t_comp = time.time()
	#---------------------------compare face----------------------
	# in_compareface, out_compareface = get_io_compareface(feature, feature_truth)
	# results = await tritonClient.infer(model_name="compare_face", inputs=in_compareface, outputs=out_compareface)
	# similarity_best = results.as_numpy("similarity")[0]
	# similarity_sort_idx_best = results.as_numpy("similarity_sort_idx")[0]
	# print(feature[:,:256].shape)
	# print(feature_truth[:,:,:256].shape)
	dist = np.linalg.norm(feature[:,::1] - feature_truth[:,:,::], axis=2)
	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
	similarity = np.mean(similarity, axis=1)
	rand = np.random.random(similarity.size)
	similarity_sort_idx = np.lexsort((rand,similarity))[::-1]
	similarity_sort_idx_best = similarity_sort_idx[0]
	similarity_best = similarity[similarity_sort_idx_best]
	print("---------similarity_best: ", similarity_best)

	# infor_face = None
	# if similarity_best > 0.70:
	# 	id_faces_best = id_faces[similarity_sort_idx_best]
	# 	infor_face = redisClient.hget("FaceInfor1", id_faces_best)

	infor_face = None
	similaritys = similarity[similarity_sort_idx]
	id_faces = np.array(id_faces, dtype=np.str_)
	codes = id_faces[similarity_sort_idx][similaritys>0.75]
	print("-------codes_fail: ", codes_fail)
	print("-------codes: ", codes)
	print("-------similaritys: ", similaritys[similaritys>0.75])
	for i, code in enumerate(codes):
		if code not in codes_fail:
			similarity_best = similarity[similarity_sort_idx[i]]
			infor_face = redisClient.hget("FaceInfor1", code)
			break
	#/////////////////////////////////////////////////////////////
	print("------Duration db: ", time.time()-t_comp)
	if infor_face is None:
		name_fail_img = datetime.now().strftime('%Y-%m-%d_%H-%M')
		cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img}.jpg', img)
		return {"success": False, "error_code": 8003, "error": "Don't find any user"}
	print(infor_face)
	infor_face = infor_face.decode("utf-8").split("@@@")
	return {"success": True, "information": {"code": infor_face[0], "name": infor_face[1], "birthday": infor_face[2], "avatar": infor_face[3], "similarity": float(similarity_best)}}

@app.post("/api/spoofingCheck")
async def spoofingCheck(image: UploadFile = File(...)):
	try:
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		#---------------------------face det-------------------------
		in_retinaface, out_retinaface = get_io_retinaface(img)
		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
		croped_image = results.as_numpy("croped_image")
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		#---------------spoofing--------------
		result = SPOOFINGDET.inference([img])[0]
		print("---------result_spoofing", result)
		if result[1] > 0.85:
			# img_list = os.listdir("./image_spoofing")
			# cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img)
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		return {"success": True}
		#//////////////////////////////////////
	except Exception as e:
		return {"success": False, "error": str(e)}

@app.post("/api/compareFace")
async def compareFace(image_face: UploadFile = File(...), image_identification: UploadFile = File(...)):
	image_byte = await image_face.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	image_byte = await image_identification.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_id = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	t_det = time.time()
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(img_face)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	croped_image_face = results.as_numpy("croped_image")
	if len(croped_image_face)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
	#cv2.imwrite("abc.jpg",croped_image_face[0])
	box_face = results.as_numpy("box")[0]
	box_face = box_face.astype(int)
	w_crop = (box_face[2]-box_face[0])
	h_crop = (box_face[3]-box_face[1])

	in_retinaface, out_retinaface = get_io_retinaface(img_id)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	croped_image_id = results.as_numpy("croped_image")
	if len(croped_image_id)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}
	#---------------spoofing--------------
	box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
	result = SPOOFINGDET.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
	print("---------result_spoofing", result)
	if result[1] > 0.85:
		# img_list = os.listdir("./image_test")
		# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
		return {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
	#//////////////////////////////////////
	#////////////////////////////////////////////////////////////
	print("------Duration det: ", time.time()-t_det)
	
	t_reg = time.time()
	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image_face)
	results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature_face = results.as_numpy("feature_norm")
	feature_face = feature_face.astype(np.float16)

	in_ghostface, out_ghostface = get_io_ghostface(croped_image_id)
	results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature_id = results.as_numpy("feature_norm")
	feature_id = feature_id.astype(np.float16)
	#////////////////////////////////////////////////////////////
	print("------Duration reg: ", time.time()-t_reg)

	t_comp = time.time()
	#---------------------------compare face----------------------
	print(feature_face.shape)
	print(feature_id.shape)
	dist = np.linalg.norm(feature_face - feature_id, axis=1)
	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

	print("---------similarity: ", similarity)

	if similarity < 0.75:
		return {"success": False, "error_code": 8003, "error": "Face photo and identification photo is not similar"}
	return {"success": True, "similarity": float(similarity)}
	#/////////////////////////////////////////////////////////////

@app.post('/healthcheck')
async def health_check():
	return { 'success': True, 'message': "healthy" }

@app.post("/api/registerFacev2")
async def registerFacev2(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	try:
		code = params.code
		print(code)
		# if redisClient.hexists("FaceInfor2", code):
		# 	return {"success": False, "error_code": 8004, "error": "This user has been registered!"}

		path_avatar = f"{IMG_AVATAR}/{code}/face_0.jpg"
		path_code = os.path.join(PATH_IMG_AVATAR, code)
		# if os.path.exists(path_code):
		# 	shutil.rmtree(path_code)
		os.makedirs(path_code, exist_ok=True)

		name = params.name
		birthday = params.birthday
		imgs = []
		img_infor = []

		for i, image in enumerate(images):
			image_byte = await image.read()
			nparr = np.fromstring(image_byte, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			img_infor.append(img.shape[:2])
			# cv2.imwrite(f'{path_code}/face_{i}.jpg', img)
			img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
			imgs.append(img)
		imgs = np.array(imgs)
		img_infor = np.array(img_infor)
		#---------------------------face det-------------------------
		in_retinaface, out_retinaface = get_io_retinaface(imgs)
		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
		croped_image = results.as_numpy("croped_image")
		num_object = results.as_numpy("num_obj").squeeze(1)
		print("-----num_object: ", num_object)
		many_face_index = (np.where(num_object>1)[0]+1).tolist()
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		if len(many_face_index)!=0:
			return {"success": False, "error_code": 8007, "error": f"Too many faces in image number {str(many_face_index).strip('[]')}"}
		# print(croped_image.shape)
		# cv2.imwrite("sadas.jpg", croped_image[0])
		#////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		in_ghostface, out_ghostface = get_io_ghostface(croped_image)
		results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
		feature = results.as_numpy("feature_norm")
		feature = feature.astype(np.float16)
		print(feature.shape)

		# dt_person_information = {code: f"{code},./{name},./{birthday}"}
		# dt_person_feature = {code: feature.tobytes()}
		miss_det = (np.where(num_object<1)[0]).tolist()
		[imgs.pop(idx) for idx in reversed(sorted(miss_det))]

		id_faces = redisClient.hkeys("FaceListCode2")
		print(len(id_faces))
		if len(id_faces)!=0:
			id_faces = int(id_faces[-1]) + 1
		else:
			id_faces = 0

		redisClient.hset("FaceInfor2", code, f"{code}@@@{name}@@@{birthday}@@@{path_avatar}")
		for i, ft in enumerate(feature):
			num_face = len(os.listdir(path_code))
			cv2.imwrite(f'{path_code}/face_{num_face}.jpg', imgs[i])
			redisClient.hset("FaceFeature2", f"{id_faces+i}", ft.tobytes())
			redisClient.hset("FaceListCode2", f"{id_faces+i}", f"{code}")

		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteAllUserv2")
def deleteAllUserv2():
	try:
		redisClient.delete("FaceInfor2")
		redisClient.delete("FaceFeature2")
		redisClient.delete("FaceListCode2")
		if os.path.exists(PATH_IMG_AVATAR):
			shutil.rmtree(PATH_IMG_AVATAR)
			os.mkdir(PATH_IMG_AVATAR)
		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteUserv2")
def deleteUserv2(codes: List[str] = ["001099008839"]):
	print(codes)
	try:
		codes_noregister = []
		for code in codes:
			if not redisClient.hexists("FaceInfor2", code):
				codes_noregister.append(code)
		if len(codes_noregister)>0:
			return {"success": False, "error_code": 8006, "error": f"User {tuple(codes_noregister)} has not been registered!"}

		id_faces = np.char.decode(redisClient.hkeys("FaceListCode2"), encoding="utf-8")
		code_list = np.char.decode(redisClient.hvals("FaceListCode2"), encoding="utf-8")
		# print(code_list)
		# print(id_faces)
		code_idx = np.where(codes[0] == code_list)
		redisClient.hdel("FaceFeature2", *id_faces[code_idx[0].tolist()])
		redisClient.hdel("FaceListCode2", *id_faces[code_idx[0].tolist()])
		redisClient.hdel("FaceInfor2", *codes)

		for code in codes:
			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/getInformationUserv2")
def getInformationUserv2(codes: List[str] = []):
	try:
		infor_persons = {}
		print(codes)
		if len(codes)==0:
			key_infor_persons = redisClient.hkeys("FaceInfor2")
			if len(key_infor_persons)==0:
				return {"success": True, "information": infor_persons}
			key_infor_persons = b'-;'.join(key_infor_persons).decode('utf-8').split("-;")
			val_infor_persons = redisClient.hvals("FaceInfor2")
			val_infor_persons = np.array(b'@@@'.join(val_infor_persons).decode('utf-8').split("@@@")).reshape(-1,4)	# shape (-1,3) for 3 field: code, name, birthday
			infor_persons = dict(zip(key_infor_persons, val_infor_persons.tolist()))
		else:
			for code in codes:
				print(redisClient.hexists("FaceInfor2", code))
				if not redisClient.hexists("FaceInfor2", code):
					infor_persons[code] = "No register"
					continue
				infor_person = redisClient.hget("FaceInfor2", code)
				infor_person = infor_person.decode("utf-8").split("@@@")
				infor_persons[code] = {"id": infor_person[0], \
										"name": infor_person[1], \
										"birthday": infor_person[2], \
										"avatar": infor_person[3]
										}
		return {"success": True, "information": infor_persons}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/searchUserv2")
async def searchUserv2(image: UploadFile = File(...)):
	id_faces = redisClient.hkeys("FaceInfor2")
	if len(id_faces) == 0:
		return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}
	image_byte = await image.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	t_det = time.time()
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(img)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	croped_image = results.as_numpy("croped_image")
	if len(croped_image)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face"}

	box = results.as_numpy("box")[0]
	# print((box[2]-box[0])*(box[3]-box[1]))
	area_img = img.shape[0]*img.shape[1]
	w_crop = (box[2]-box[0])
	h_crop = (box[3]-box[1])
	# if not area_img*0.15<w_crop*h_crop<area_img*0.3:
	# 	return {"success": False, "error_code": 8009, "error": "Face size is not true"}
	#---------------spoofing--------------
	box_expand = np.array([max(box[0]-w_crop,0), max(box[1]-h_crop,0), min(box[2]+w_crop, img.shape[1]), min(box[3]+h_crop, img.shape[0])], dtype=int)
	result = SPOOFINGDET.inference([img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
	# result = SPOOFINGDET.inference([img])[0]
	print("---------result_spoofing", result)
	if result[1] > 0.78:
		img_list = os.listdir("./image_spoofing")
		cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
		return {"success": False, "error_code": 8002, "error": "Fake face image"}
	#//////////////////////////////////////
	#////////////////////////////////////////////////////////////
	print("------Duration det: ", time.time()-t_det)

	t_reg = time.time()
	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	#////////////////////////////////////////////////////////////
	print("------Duration reg: ", time.time()-t_reg)

	# #---------------------------compare fail face---------------------
	# ft_faces = np.array(redisClient.hvals("FaceFeatureFail2"))
	# if len(ft_faces)==0:
	# 	codes_fail = []
	# else:
		# feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)
		# # print(feature_truth.shape)
		# # print(feature.shape)
		# dist = np.linalg.norm(feature - feature_truth, axis=1)
		# similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

		# ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeatureFail2"), encoding="utf-8").astype(int)
		# code_list = np.char.decode(redisClient.hvals("FaceListCodeFail2"), encoding="utf-8")

		# codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
		# ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
		# similarity = similarity[ft_faces_idx_sort]				# value with sorted key
		# similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
		# # print(similarity_average)
		# rand = np.random.random(similarity_average.size)
		# idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
		# print(idx_sorted)
		# similaritys = similarity_average[idx_sorted]
		# codes_fail = codes[idx_sorted][similaritys>0.8]
	# #/////////////////////////////////////////////////////////////////

	t_db = time.time()
	ft_faces = np.array(redisClient.hvals("FaceFeature2"))
	feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)
	print("------Duration db: ", time.time()-t_db)

	t_comp = time.time()
	#---------------------------compare face----------------------
	print(feature_truth.shape)
	print(feature.shape)
	dist = np.linalg.norm(feature - feature_truth, axis=1)
	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
	# print(similarity)

	ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeature2"), encoding="utf-8").astype(int)
	code_list = np.char.decode(redisClient.hvals("FaceListCode2"), encoding="utf-8")

	codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
	ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
	similarity = similarity[ft_faces_idx_sort]				# value with sorted key
	similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
	# print(similarity_average)
	rand = np.random.random(similarity_average.size)
	idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
	similarity_best = similarity_average[idx_sorted[0]]
	print("---------similarity_best: ", similarity_best)

	infor_face = None
	if similarity_best > 0.75:
		code = codes[idx_sorted[0]]
		infor_face = redisClient.hget("FaceInfor2", code)

	# infor_face = None
	# similaritys = similarity_average[idx_sorted]
	# codes = codes[idx_sorted][similaritys>0.75]
	# for i, code in enumerate(codes):
	# 	if code not in codes_fail:
	# 		similarity_best = similarity_average[idx_sorted[i]]
	# 		infor_face = redisClient.hget("FaceInfor2", code)
			# break
	#/////////////////////////////////////////////////////////////
	print("------Duration db: ", time.time()-t_comp)
	if infor_face is None:
		return {"success": False, "error_code": 8003, "error": "Don't find any user"}
	print(infor_face)
	infor_face = infor_face.decode("utf-8").split("@@@")
	return {"success": True, "information": {"code": infor_face[0], "name": infor_face[1], "birthday": infor_face[2], "avatar": infor_face[3], "similarity": float(similarity_best)}}

@app.post("/api/checkFailFacev2")
async def checkFailFacev2(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	# try:
	code = params.code
	print(code)

	name = params.name
	birthday = params.birthday
	imgs = []

	num_img = len(os.listdir(PATH_IMG_FAIL))
	for i, image in enumerate(images):
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		cv2.imwrite(f'{PATH_IMG_FAIL}/{code}_{name}_{num_img+i}.jpg', img)
		img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
		imgs.append(img)
	imgs = np.array(imgs)
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(imgs)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	croped_image = results.as_numpy("croped_image")
	num_object = results.as_numpy("num_obj").squeeze(1)

	if len(croped_image)==0:
		return {"success": True}
	#////////////////////////////////////////////////////////////

	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	print(feature.shape)

	miss_det = (np.where(num_object<1)[0]).tolist()
	[imgs.pop(idx) for idx in reversed(sorted(miss_det))]

	id_faces = redisClient.hkeys("FaceListCodeFail2")
	print(len(id_faces))
	if len(id_faces)!=0:
		id_faces = int(id_faces[-1]) + 1
	else:
		id_faces = 0

	for i, ft in enumerate(feature):
		redisClient.hset("FaceFeatureFail2", f"{id_faces+i}", ft.tobytes())
		redisClient.hset("FaceListCodeFail2", f"{id_faces+i}", f"{code}")

	return {"success": True}
	# except Exception as e:
	# 	return {"success": False, "error_code": 8008, "error": str(e)}

# @app.post("/api/getFailFacev2")
# async def getFailFacev2(image: UploadFile = File(...)):
# 	image_byte = await image.read()
# 	nparr = np.fromstring(image_byte, np.uint8)
# 	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# 	#---------------------------face det-------------------------
# 	in_retinaface, out_retinaface = get_io_retinaface(img)
# 	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
# 	croped_image = results.as_numpy("croped_image")
# 	if len(croped_image)==0:
# 		return {"success": False, "error_code": 8001, "error": "Don't find any face"}

# 	#---------------------------face reg-------------------------
# 	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
# 	results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
# 	feature = results.as_numpy("feature_norm")
# 	feature = feature.astype(np.float16)
# 	#////////////////////////////////////////////////////////////

# 	ft_faces = np.array(redisClient.hvals("FaceFeatureFail2"))
# 	feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)

# 	print(feature_truth.shape)
# 	print(feature.shape)
# 	dist = np.linalg.norm(feature - feature_truth, axis=1)
# 	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

# 	ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeatureFail2"), encoding="utf-8").astype(int)
# 	code_list = np.char.decode(redisClient.hvals("FaceListCodeFail2"), encoding="utf-8")

# 	codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
# 	ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
# 	similarity = similarity[ft_faces_idx_sort]				# value with sorted key
# 	similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
# 	# print(similarity_average)
# 	rand = np.random.random(similarity_average.size)
# 	idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
# 	print(idx_sorted)
# 	similaritys = similarity_average[idx_sorted]
# 	codes = codes[idx_sorted][similaritys>0.8]

# 	print("-----idx_sorted: ", idx_sorted)
# 	print(similaritys)
# 	print(codes)
# 	print("001099008838" in codes)


@app.post("/api/deleteFailFacev2")
async def deleteFailFacev2():
	try:
		redisClient.delete("FaceFeatureFail2")
		redisClient.delete("FaceListCodeFail2")
		if os.path.exists(PATH_IMG_FAIL):
			shutil.rmtree(PATH_IMG_FAIL)
			os.mkdir(PATH_IMG_FAIL)
		return {"success": True}
	except Exception as e:
		return {"success": False, "error_code": 8008, "error": str(e)}

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8421

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=True)


"""
8000: "Don't have any registered user"
8001: "Don't find any face"
8002: "Fake face image"
8003: "Don't find any user"
8004: "This user has been registered!"
8005: "No users have been registered!"
8006: "This user has not been registered!"
8007: "Too many faces in this image"
8008: error system
8009: "Face size is not true"
8010: "There are some special letters in user code!"
"""


# docker run -it --shm-size=4g --rm -p8000:8000 -p8001:8001 -p8002:8002 -e PYTHONIOENCODING=UTF-8 -v ${PWD}:/workspace/ -v ${PWD}/my_repository:/models -v ${PWD}/requirements.txt:/opt/tritonserver/requirements.tx tritonserver_mq

# tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5
