from typing import Any
import cv2
import torch
from tqdm import tqdm
import numpy as np


from src.DECA.decalib.deca import DECA
from src.DECA.decalib.datasets import datasets 
from src.DECA.decalib.utils import util
from src.DECA.decalib.utils.config import cfg as deca_cfg
from src.DECA.decalib.utils.tensor_cropper import transform_points

class LMKExtractor():

	def __init__(self) -> None:
		pass

	def __call__(open_cv_image) -> Any:
		# load test images 
		testdata = datasets.TestData(open_cv_image, iscrop=True, face_detector='fan', sample_step=10)

		# run DECA
		deca_cfg.model.use_tex = False
		deca_cfg.rasterizer_type = 'pytorch3d'
		deca_cfg.model.extract_tex = True
		deca = DECA(config = deca_cfg, device='cuda')
		# for i in range(len(testdata)):
		for i in tqdm(range(len(testdata))):
			name = testdata[i]['imagename']
			images = testdata[i]['image'].to('cuda')[None,...]
			with torch.no_grad():
				codedict = deca.encode(images)
				opdict, visdict = deca.decode(codedict) #tensor
				#render_org in default
				tform = testdata[i]['tform'][None, ...]
				tform = torch.inverse(tform).transpose(1,2).to('cuda')
				original_image = testdata[i]['original_image'][None, ...].to('cuda')
				_, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
				orig_visdict['inputs'] = original_image 
				landmark_51 = opdict['landmarks3d_world'][:, 17:]
				landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
				landmark_7 = landmark_7.cpu().numpy() 

				lmks = np.array(opdict['landmarks3d_world'])
				
				lmks3d = np.array(opdict['verts'])
				lmks3d = lmks3d.reshape(-1, 5)[:, :3]
				mp_tris = np.array(opdict['verts'].index).reshape(-1, 3) + 1

				return {
					"lmks": lmks,
					'lmks3d': lmks3d,
					"trans_mat": opdict['trans_verts'],
					'faces': mp_tris,
				}  