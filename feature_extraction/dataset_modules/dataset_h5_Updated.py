import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None,
		target_magnification=20):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			wsi: OpenSlide WSI object
			img_transforms (callable, optional): Optional transform to be applied on a sample
			target_magnification (int): Target magnification for downsampling (default: 20)
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms
		self.target_magnification = target_magnification

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		# Calculate magnification info
		self._calculate_magnification_info()
		self.summary()
			
	def _calculate_magnification_info(self):
		"""Calculate magnification and downsampling parameters"""
		# Get the magnification of the slide at level 0 (highest resolution)
		try:
			# Try to get magnification from slide properties
			if 'aperio.AppMag' in self.wsi.properties:
				self.base_magnification = float(self.wsi.properties['aperio.AppMag'])
			elif 'openslide.objective-power' in self.wsi.properties:
				self.base_magnification = float(self.wsi.properties['openslide.objective-power'])
			else:
				# Default assumption - most slides are scanned at 40x
				self.base_magnification = 40.0
				print(f"Warning: Could not determine slide magnification, assuming {self.base_magnification}x")
		except:
			self.base_magnification = 40.0
			print(f"Warning: Could not determine slide magnification, assuming {self.base_magnification}x")
		
		# Calculate current magnification at the patch level
		level_downsample = self.wsi.level_downsamples[self.patch_level]
		self.current_magnification = self.base_magnification / level_downsample
		
		# Calculate additional downsampling needed to reach target magnification
		if self.current_magnification > self.target_magnification:
			self.downsample_factor = self.current_magnification / self.target_magnification
			self.needs_downsampling = True
		else:
			self.downsample_factor = 1.0
			self.needs_downsampling = False
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)
		print(f'base magnification: {self.base_magnification}x')
		print(f'current magnification at level {self.patch_level}: {self.current_magnification}x')
		print(f'target magnification: {self.target_magnification}x')
		print(f'downsample factor: {self.downsample_factor}x')
		print(f'needs downsampling: {self.needs_downsampling}')

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		
		# Read the patch at the original resolution
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		
		# Apply manual downsampling if needed
		if self.needs_downsampling:
			# Calculate new size after downsampling
			new_size = int(self.patch_size / self.downsample_factor)
			# Use high-quality resampling (Lanczos) for downsampling
			img = img.resize((new_size, new_size), Image.LANCZOS)
			
			# If the target patch size is needed for the model, resize back up
			# This step depends on your model requirements
			if hasattr(self, 'roi_transforms') and self.roi_transforms is not None:
				# Let the transforms handle the final resizing
				pass
			else:
				# If no transforms, you might want to resize to a standard size
				# img = img.resize((224, 224), Image.LANCZOS)  # Uncomment if needed
				pass

		# Apply transformations
		if self.roi_transforms:
			img = self.roi_transforms(img)
			
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]