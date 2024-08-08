# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from math import ceil
from itertools import product as product
import cv2
from face_preprocess import preprocess as face_preprocess
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from PIL import Image

class PriorBox(object):
	def __init__(self, min_sizes=[[16,32],[64,128],[256,512]], steps=[8,16,32], clip=False, image_size=None, phase='train'):
		super(PriorBox, self).__init__()
		self.min_sizes = min_sizes
		self.steps = steps
		self.clip = clip
		self.image_size = image_size
		self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
		self.name = "s"

	def forward(self):
		anchors = []
		for k, f in enumerate(self.feature_maps):
			min_sizes = self.min_sizes[k]
			for i, j in product(range(f[0]), range(f[1])):
				for min_size in min_sizes:
					s_kx = min_size / self.image_size[1]
					s_ky = min_size / self.image_size[0]
					dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
					dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
					for cy, cx in product(dense_cy, dense_cx):
						anchors += [cx, cy, s_kx, s_ky]

		# back to torch land
		output = torch.Tensor(anchors).view(-1, 4)
		if self.clip:
			output.clamp_(max=1, min=0)
		return output

class TritonPythonModel:
	"""Your Python model must use the same class name. Every Python model
	that is created must have "TritonPythonModel" as the class name.
	"""

	def initialize(self, args):
		"""`initialize` is called only once when the model is being loaded.
		Implementing `initialize` function is optional. This function allows
		the model to initialize any state associated with this model.
		Parameters
		----------
		args : dict
		  Both keys and values are strings. The dictionary keys and values are:
		  * model_config: A JSON string containing the model configuration
		  * model_instance_kind: A string containing model instance kind
		  * model_instance_device_id: A string containing model instance device ID
		  * model_repository: Model repository path
		  * model_version: Model version
		  * model_name: Model name
		"""

		# You must parse model_config. JSON string is not parsed here
		model_config = json.loads(args["model_config"])

		# Get OUTPUT1 configuration
		classes_config = pb_utils.get_output_config_by_name(
			model_config, "detection_retinaface_postprocessing_classes"
		)
		numobject_config = pb_utils.get_output_config_by_name(
			model_config, "detection_retinaface_postprocessing_numobject"
		)
		croped_image_config = pb_utils.get_output_config_by_name(
			model_config, "croped_image"
		)
		box_config = pb_utils.get_output_config_by_name(
			model_config, "detection_retinaface_postprocessing_box"
		)

		# Convert Triton types to numpy types
		self.classes_dtype = pb_utils.triton_string_to_numpy(
			classes_config["data_type"]
		)
		# Convert Triton types to numpy types
		self.croped_image_dtype = pb_utils.triton_string_to_numpy(
			croped_image_config["data_type"]
		)
		self.numobject_dtype = pb_utils.triton_string_to_numpy(
			numobject_config["data_type"]
		)
		self.box_dtype = pb_utils.triton_string_to_numpy(
			box_config["data_type"]
		)
		self.min_sizes = [[16,32], [64,128], [256,512]]
		self.steps = [8, 16, 32]
		self.variance = [0.1, 0.2]
		self.clip = False
		self.conf_thres = 0.75
		self.iou_thres = 0.25
		self.imagesz = 640

	def py_cpu_nms(self, dets, thresh):
		x1 = dets[:, 0]
		y1 = dets[:, 1]
		x2 = dets[:, 2]
		y2 = dets[:, 3]
		scores = dets[:, 4]

		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= thresh)[0]
			order = order[inds + 1]

		return keep


	def decode(self, loc, priors, variances):
		boxes = torch.cat((
			priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
			priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
		boxes[:, :2] -= boxes[:, 2:] / 2
		boxes[:, 2:] += boxes[:, :2]
		return boxes

	def decode_landm(self, pre, priors, variances):
		landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
							), dim=1)
		return landms

	def postProcess(self, img_info, loc, conf, landms):
			loc = torch.Tensor(loc)
			conf = torch.Tensor(conf)
			landms = torch.Tensor(landms)
			im_height, im_width = img_info
			scale = np.array([im_width, im_height, im_width, im_height])
			priorbox = PriorBox(min_sizes=self.min_sizes, steps=self.steps, \
							clip=self.clip, image_size=(self.imagesz, self.imagesz))
			priors = priorbox.forward()
			prior_data = priors.data
			boxes = self.decode(loc.data, prior_data, self.variance)
			boxes = boxes * scale
			boxes = boxes.cpu().numpy()
			scores = conf.data.cpu().numpy()[:, 1]
			landms = self.decode_landm(landms.data, prior_data, self.variance)
			scale_landms = torch.Tensor([im_width, im_height, im_width, im_height, im_width,
								   im_height, im_width, im_height, im_width, im_height])
			landms = landms * scale_landms
			landms = landms.cpu().numpy()

			inds = np.where(scores > self.conf_thres)[0]
			boxes = boxes[inds]
			landms = landms[inds]
			scores = scores[inds]

			order = scores.argsort()[::-1][:5000]
			boxes = boxes[order]
			landms = landms[order]
			scores = scores[order]

			dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
			keep = self.py_cpu_nms(dets, self.iou_thres)
			dets = dets[keep, :]
			landms = landms[keep]

			dets = dets[:750, :]
			landms = landms[:750, :]
			dets = np.concatenate((dets, landms), axis=1)
			return dets

	def execute(self, requests):
		"""`execute` MUST be implemented in every Python model. `execute`
		function receives a list of pb_utils.InferenceRequest as the only
		argument. This function is called when an inference request is made
		for this model. Depending on the batching configuration (e.g. Dynamic
		Batching) used, `requests` may contain multiple requests. Every
		Python model, must create one pb_utils.InferenceResponse for every
		pb_utils.InferenceRequest in `requests`. If there is an error, you can
		set the error argument when creating a pb_utils.InferenceResponse
		Parameters
		----------
		requests : list
		  A list of pb_utils.InferenceRequest
		Returns
		-------
		list
		  A list of pb_utils.InferenceResponse. The length of this list must
		  be the same as `requests`
		"""

		classes_dtype = self.classes_dtype
		croped_image_dtype = self.croped_image_dtype
		numobject_dtype = self.numobject_dtype
		box_dtype = self.box_dtype

		responses = []

		# Every Python backend must iterate over everyone of the requests
		# and create a pb_utils.InferenceResponse for each of them.
		for request in requests:
			# Get INPUT0
			in_info = pb_utils.get_input_tensor_by_name(
				request, "detection_retinaface_postprocessing_input_info"
			)
			in_det = pb_utils.get_input_tensor_by_name(
				request, "detection_retinaface_postprocessing_input_det"
			)
			in_cls = pb_utils.get_input_tensor_by_name(
				request, "detection_retinaface_postprocessing_input_cls"
			)
			in_landmark = pb_utils.get_input_tensor_by_name(
				request, "detection_retinaface_postprocessing_input_landmark"
			)
			in_img = pb_utils.get_input_tensor_by_name(
				request, "detection_retinaface_postprocessing_input_image"
			)

			# out_dets = np.empty((0,5), float)
			# out_classes = np.empty((0,2), float)
			# out_landmarks = np.empty((0,11), float)
			croped_images = []
			out_classes = []
			num_objects = []
			boxs = []
			for i, img in enumerate(in_img.as_numpy()):
				img_info = in_info.as_numpy()[i]
				loc = in_det.as_numpy()[i]
				conf = in_cls.as_numpy()[i]
				landms = in_landmark.as_numpy()[i]

				dets = self.postProcess(img_info, loc, conf, landms)

				bboxes = dets[:,:4]
				classes = dets[:,4]
				landms = dets[:,5:]

				num_objects.append([len(bboxes)])

				#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				biggestBox = None
				maxArea = 0
				#nimg = np.empty(0)
				for j, bbox in enumerate(bboxes):
					x1, y1, x2, y2 = bbox
					area = (x2-x1) * (y2-y1)
					if area > maxArea:
						maxArea = area
						biggestBox = bbox
						landmarks = landms[j]
						
				if biggestBox is not None:
					out_classes.extend(np.expand_dims(classes, axis=0))
					bbox = np.array(biggestBox)
					boxs.append(bbox)
					landmarks = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
								landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
					landmarks = landmarks.reshape((2,5)).T
					nimg = face_preprocess(img, bbox, landmarks, image_size=[112,112])
					croped_images.extend(np.expand_dims(nimg, axis=0))
			out_classes = np.array(out_classes)
			croped_images = np.array(croped_images)
			num_objects = np.array(num_objects)
			boxs = np.array(boxs)
			# print(num_objects)
			# print(numobject_dtype)
			# 	img_id = np.broadcast_to(i, (len(dets),1))
			# 	out_dets = np.vstack( (out_dets, np.append(dets[:,:4], img_id, axis=1)) )	# out[n,4] -> out[n,5] -> out[n,n,5] with out[:,:,-1] is "id" of image
			# 	out_classes = np.vstack( (out_classes, np.append(dets[:,4][:,None], img_id, axis=1)) )
			# 	out_landmarks = np.vstack( (out_landmarks, np.append(dets[:,5:], img_id, axis=1)) )

			# out_dets = np.array(out_dets).reshape(-1, 5)
			# out_classes = np.array(out_classes).reshape(-1, 2)
			# out_landmarks = np.array(out_landmarks).reshape(-1, 11)

			# img_info = in_info.as_numpy()[0]
			# loc = in_det.as_numpy()
			# conf = in_cls.as_numpy()
			# landms = in_landmark.as_numpy()

			# dets = self.postProcess(img_info, loc, conf, landms)
			# out_dets = np.expand_dims(np.array(dets[:,:4]), axis=0)
			# out_classes = np.expand_dims(np.array(dets[:,4]), axis=0)
			# out_landmarks = np.expand_dims(np.array(dets[:,5:]), axis=0)

			out_tensor_classes = pb_utils.Tensor(
				"detection_retinaface_postprocessing_classes", out_classes.astype(classes_dtype)
			)
			out_tensor_croped_image = pb_utils.Tensor(
				"croped_image", croped_images.astype(croped_image_dtype)
			)
			out_tensor_numobject = pb_utils.Tensor(
				"detection_retinaface_postprocessing_numobject", num_objects.astype(numobject_dtype)
			)
			out_tensor_box = pb_utils.Tensor(
				"detection_retinaface_postprocessing_box", boxs.astype(box_dtype)
			)

			# Create InferenceResponse. You can set an error here in case
			# there was a problem with handling this inference request.
			# Below is an example of how you can set errors in inference
			# response:
			#
			# pb_utils.InferenceResponse(
			#    output_tensors=..., TritonError("An error occurred"))
			inference_response = pb_utils.InferenceResponse(
				output_tensors=[out_tensor_croped_image, out_tensor_classes, out_tensor_numobject, out_tensor_box]
			)
			responses.append(inference_response)
		# You should return a list of pb_utils.InferenceResponse. Length
		# of this list must match the length of `requests` list.
		return responses

	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is OPTIONAL. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print("Cleaning up...")
