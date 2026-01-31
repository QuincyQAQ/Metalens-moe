import os
import cv2
import glob
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode

from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img


class CDD11(Dataset):
    def __init__(self, args, split: str = "train", subset: str = "all"):
        super(CDD11, self).__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.dataset_split = split
        self.subset = subset
        if split == "train":
            self.patch_size = args.patch_size
        else:
            self.patch_size = 64

        self._init()

    def __getitem__(self, index):
        # Randomly select a degradation type
        if self.dataset_split == "train":
            degradation_type = random.choice(list(self.degraded_dict.keys()))
            degraded_image_path = random.choice(self.degraded_dict[degradation_type])
        else:
            degradation_type = self.subset
            degraded_image_path = self.degraded_dict[degradation_type][index]
        
        # Select a degraded image within that type

        degraded_name = os.path.basename(degraded_image_path)

        # Get the corresponding clean image based on the file name
        image_name = os.path.basename(degraded_image_path)
        assert degraded_name == image_name
        clean_image_path = os.path.join(os.path.dirname(self.clean[0]), image_name)

        # Load the images
        #lr = crop_img(np.array(Image.open(degraded_image_path).convert('RGB')), base=16)
        lr = np.array(Image.open(degraded_image_path).convert('RGB'))
        #hr = crop_img(np.array(Image.open(clean_image_path).convert('RGB')), base=16)
        hr = np.array(Image.open(clean_image_path).convert('RGB'))
        # Apply random augmentation and crop
        if self.dataset_split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # Convert to tensors
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [clean_image_path, degradation_type], lr, hr

    def __len__(self):
        return sum(len(images) for images in self.degraded_dict.values())

    def _init(self):
        data_dir = os.path.join(self.args.data_file_dir, "cdd11")
        self.clean = sorted(glob.glob(os.path.join(data_dir, f"{self.dataset_split}/clear", "*.png")))

        if len(self.clean) == 0:
            raise ValueError(f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        self.degraded_dict = {}
        allowed_degradation_folders = self._filter_degradation_folders(data_dir)
        for folder in allowed_degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            degraded_images = sorted(glob.glob(os.path.join(folder, "*.png")))
            
            if len(degraded_images) == 0:
                raise ValueError(f"No images found in {folder_name}")
            
            # Removed artificial repetition multiplier (*2) for faster epoch completion
            
            self.degraded_dict[folder_name] = degraded_images

    def _filter_degradation_folders(self, data_dir):
        """
        This function returns folders based on the degradation_type_mode.
        'single', 'double', 'triple', or 'all' degradation types will be returned.
        """
        degradation_folders = sorted(glob.glob(os.path.join(data_dir, self.dataset_split, "*/")))
        filtered_folders = [] 

        for folder in degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            if folder_name == "clear":
                continue

            # Count the number of degradations based on the number of underscores in the folder name
            degradation_count = folder_name.count('_') + 1

            # Check the degradation type mode and filter accordingly
            if self.subset == "single" and degradation_count == 1:
                filtered_folders.append(folder)
            elif self.subset == "double" and degradation_count == 2:
                filtered_folders.append(folder)
            elif self.subset == "triple" and degradation_count == 3:
                filtered_folders.append(folder)
            elif self.subset == "all":
                filtered_folders.append(folder)
            # If self.subset is a specific degradation folder name, match it exactly
            elif self.subset not in ["single", "double", "triple", "all"]:
                if folder_name == self.subset:
                    filtered_folders.append(folder)

        print(f"Degradation type mode: {self.subset}")
        print(f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders

    def _crop_patch(self, img_1, img_2):
        # Crop a patch from both images (degraded and clean) at the same location
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
    
        
class AIOTrainDataset(Dataset):
    """
    Dataset class for training on degraded images.
    """
    def __init__(self, args):
        super(AIOTrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}
        
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        self._init_lr()
        self._merge_tasks()
            
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]
        
        if deg_type == "denoise_15" or deg_type == "denoise_25" or deg_type == "denoise_50":
            # 优化：直接从 PIL Image 裁剪 patch，避免加载整张图
            hr_img = Image.open(lr_sample["img"]).convert('RGB')
            hr = self.crop_transform(np.array(hr_img))
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, de_id)
        else:
            if deg_type == "dehaze":
                # 优化：直接从 PIL Image 裁剪 patch
                lr_img = Image.open(lr_sample["img"]).convert('RGB')
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr_img = Image.open(clean_name).convert('RGB')
                
                # 随机选择相同的裁剪位置
                lr_arr = np.array(lr_img)
                hr_arr = np.array(hr_img)
                lr, hr = random_augmentation(*self._crop_patch(lr_arr, hr_arr))
            else:
                # 优化：直接从 PIL Image 裁剪 patch，避免 crop_img 对齐操作
                hr_sample = self.hr[idx]
                lr_img = Image.open(lr_sample["img"]).convert('RGB')
                hr_img = Image.open(hr_sample["img"]).convert('RGB')
                
                # 随机选择相同的裁剪位置
                lr_arr = np.array(lr_img)
                hr_arr = np.array(hr_img)
                lr, hr = random_augmentation(*self._crop_patch(lr_arr, hr_arr))
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        
        return [lr_sample["img"], de_id], lr, hr
        
    
    def __len__(self):
        return len(self.lr)
    
    
    def _init_lr(self):
        # synthetic datasets
        if 'synllie' in self.de_type:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'deblur' in self.de_type:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'derain' in self.de_type:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.de_type:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_25' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_50' in self.de_type:
            self._init_clean(id=0)
            
    def _merge_tasks(self):
        self.lr = []
        self.hr = []
        # synthetic datasets
        if "synllie" in self.de_type:
            self.lr += self.synllie_lr
            self.hr += self.synllie_hr
        if "denoise_15" in self.de_type:
            self.lr += self.s15_ids
            self.hr += self.s15_ids
        if "denoise_25" in self.de_type:
            self.lr += self.s25_ids
            self.hr += self.s25_ids
        if "denoise_50" in self.de_type:
            self.lr += self.s50_ids
            self.hr += self.s50_ids
        if "deblur" in self.de_type:
            self.lr += self.deblur_lr 
            self.hr += self.deblur_hr
        if "derain" in self.de_type:
            self.lr += self.derain_lr 
            self.hr += self.derain_hr
        if "dehaze" in self.de_type:
            self.lr += self.dehaze_lr 
            self.hr += self.dehaze_hr

        print(len(self.lr))
   
            
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/llie/LOLv1/Train/input"
        targets = self.args.data_file_dir + "/llie/LOLv1/Train/target"
        
        self.synllie_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.synllie_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.synllie_counter = 0
        print("Total SynLLIE training pairs : {}".format(len(self.synllie_lr)))
        # Removed artificial repetition multiplier (*20) for faster epoch completion
        print("Dataset length : {}".format(len(self.synllie_hr)))
    
    def _init_deblur(self, id):
        """Initialize the deblur training dataset.

        优先级：
        1) Metalens_0622 结构:  <data_file_dir>/Metalens_0622/train/meta, ground_truth
        2) 通用结构:            <data_file_dir>/train/lr, gt   （例如 open_dataset_8_1_1）
        3) GoPro 结构:          <data_file_dir>/deblurring/GoPro/crop/train/input_crops, target_crops
        """

        # 1) Metalens_0622
        metalens_inputs = os.path.join(self.args.data_file_dir, "Metalens_0622", "train", "meta")
        metalens_targets = os.path.join(self.args.data_file_dir, "Metalens_0622", "train", "ground_truth")

        # 2) 通用 train/lr, train/gt 结构（例如 open_dataset_8_1_1）
        generic_inputs = os.path.join(self.args.data_file_dir, "train", "lr")
        generic_targets = os.path.join(self.args.data_file_dir, "train", "gt")

        if os.path.exists(metalens_inputs) and os.path.exists(metalens_targets):
            self.deblur_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(metalens_inputs, "*.png")))]
            self.deblur_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(metalens_targets, "*.png")))]
            print("Using Metalens_0622 dataset structure")

        elif os.path.exists(generic_inputs) and os.path.exists(generic_targets):
            # 直接使用 <root>/train/lr 与 <root>/train/gt
            self.deblur_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(generic_inputs, "*.png")))]
            self.deblur_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(generic_targets, "*.png")))]
            print("Using generic train/lr & train/gt dataset structure")

        else:
            # 3) 回退到 GoPro 数据集结构
            inputs = os.path.join(self.args.data_file_dir, "deblurring", "GoPro", "crop", "train", "input_crops")
            targets = os.path.join(self.args.data_file_dir, "deblurring", "GoPro", "crop", "train", "target_crops")
            self.deblur_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(inputs, "*.png")))]
            self.deblur_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(targets, "*.png")))]
            print("Using GoPro dataset structure")

        self.deblur_counter = 0
        print("Total Deblur training pairs : {}".format(len(self.deblur_hr)))
        # Removed artificial repetition multiplier (*5) for faster epoch completion
        print("Dataset length : {}".format(len(self.deblur_hr)))
        
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/deraining/RainTrainL/rainy"
        targets = self.args.data_file_dir + "/deraining/RainTrainL/gt"
        
        self.derain_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.derain_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.derain_counter = 0
        print("Total Derain training pairs : {}".format(len(self.derain_lr)))
        # Removed artificial repetition multiplier (*120) for faster epoch completion
        print("Dataset length : {}".format(len(self.derain_hr)))
        
    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/dehazing/RESIDE/"
        targets = self.args.data_file_dir + "/dehazing/RESIDE/clear"
        
        self.dehaze_lr = []
        for part in ["part1", "part2", "part3", "part4"]:
            self.dehaze_lr += [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + part + "/*.jpg"))]
        
        self.dehaze_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.jpg"))]
        
        self.dehaze_counter = 0
        print("Total Dehaze training pairs : {}".format(len(self.dehaze_lr)))
        self.dehaze_lr = self.dehaze_lr
        self.dehaze_hr = self.dehaze_hr
        print("Repeated Dataset length : {}".format(len(self.dehaze_lr)))
        
    def _init_clean(self, id):
        inputs = self.args.data_file_dir + "/denoising"
        
        clean = []
        for dataset in ["WaterlooED", "BSD400"]:
            if dataset == "WaterlooED":
                ext = "bmp"
            else:
                ext = "jpg"
            clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]
            
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"img": x, "de_type":self.de_dict['denoise_15']} for x in clean]
            # Removed artificial repetition multiplier (*3) for faster epoch completion
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"img": x, "de_type":self.de_dict['denoise_25']} for x in clean]
            # Removed artificial repetition multiplier (*3) for faster epoch completion
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"img": x, "de_type":self.de_dict['denoise_50']} for x in clean]
            # Removed artificial repetition multiplier (*3) for faster epoch completion
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/clear"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + suffix
        return nonhazy_name
        
    
class IRBenchmarks(Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()
        
        self.args = args
        self.benchmarks = args.benchmarks
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        # 验证/测试阶段也使用 patch 逻辑，patch_size 与训练保持一致
        self.patch_size = getattr(self.args, "patch_size", 128)
        self.full_res_eval = bool(getattr(self.args, "full_res_eval", False))

        self.toTensor = ToTensor()
        
        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)
        
        self._init_lr()
    
    def _read_image_cv2_rgb(self, path: str):
        """Read an image with OpenCV and return an RGB numpy array.

        Using OpenCV here avoids Pillow's strict MAX_TEXT_CHUNK limit on some
        PNG files with large iCCP/text chunks, which previously caused
        ValueError: Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK.
        """

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _center_crop_patch(self, img):
        """验证/测试阶段使用的中心 patch 裁剪，大小与训练的 patch_size 一致。

        - 先假定输入已经过 crop_img，尺寸为 base 的倍数
        - 若图像任一边小于 patch_size，则直接返回整图（通常也不会太大）
        """

        h, w = img.shape[:2]
        p = int(self.patch_size)
        if h <= p or w <= p:
            return img

        top = (h - p) // 2
        left = (w - p) // 2
        return img[top:top + p, left:left + p, :]
        
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        
        # 和训练类似：先做 crop_img 保证尺寸是 base 的倍数，再从中取中心 patch
        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or "denoise_100" in self.benchmarks or "denoise_75" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr_np = self._read_image_cv2_rgb(lr_sample["img"])
            hr_full = crop_img(hr_np, base=16)
            hr = hr_full if self.full_res_eval else self._center_crop_patch(hr_full)
            lr, _ = self._add_gaussian_noise(hr, sigma)
        else:
            hr_sample = self.hr[idx]
            lr_np = self._read_image_cv2_rgb(lr_sample["img"])
            hr_np = self._read_image_cv2_rgb(hr_sample["img"])
            lr_full = crop_img(lr_np, base=16)
            hr_full = crop_img(hr_np, base=16)
            if self.full_res_eval:
                lr = lr_full
                hr = hr_full
            else:
                lr = self._center_crop_patch(lr_full)
                hr = self._center_crop_patch(hr_full)
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr
    
    def __len__(self):
        return len(self.lr)
    
    def _init_lr(self):
        if 'lolv1' in self.benchmarks:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'gopro' in self.benchmarks:
            self._init_deblurring("GoPro", id=self.de_dict['deblur'])
        if 'derain' in self.benchmarks:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.benchmarks:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=0)

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/gt"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + '.png'
        return nonhazy_name
    
    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
    
    ####################################################################################################
    ## DEBLURRING DATASET
    def _init_deblurring(self, benchmark, id):
        """Initialize deblurring benchmark dataset (validation/test).

        优先级：
        1) Metalens_0622 结构:  <data_file_dir>/Metalens_0622/test/meta, ground_truth
        2) 通用结构:            <data_file_dir>/test/lr, gt   （例如 open_dataset_8_1_1）
        3) GoPro 结构:          <data_file_dir>/deblurring/{benchmark}/test/input, target
        """

        # 1) Metalens_0622
        metalens_inputs = os.path.join(self.args.data_file_dir, "Metalens_0622", "test", "meta")
        metalens_targets = os.path.join(self.args.data_file_dir, "Metalens_0622", "test", "ground_truth")

        # 2) 通用 test/lr, test/gt 结构
        generic_inputs = os.path.join(self.args.data_file_dir, "test", "lr")
        generic_targets = os.path.join(self.args.data_file_dir, "test", "gt")

        if os.path.exists(metalens_inputs) and os.path.exists(metalens_targets):
            self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(metalens_inputs, "*.png")))]
            self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(metalens_targets, "*.png")))]
            print("Using Metalens_0622 test dataset structure")

        elif os.path.exists(generic_inputs) and os.path.exists(generic_targets):
            self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(generic_inputs, "*.png")))]
            self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(generic_targets, "*.png")))]
            print("Using generic test/lr & test/gt dataset structure")

        else:
            # 3) 回退到 GoPro 数据集结构
            inputs = os.path.join(self.args.data_file_dir, "deblurring", benchmark, "test", "input")
            targets = os.path.join(self.args.data_file_dir, "deblurring", benchmark, "test", "target")
            self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(inputs, "*.png")))]
            self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(os.path.join(targets, "*.png")))]
            print("Using GoPro test dataset structure")

        print("Total Deblur testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## LLIE DATASET        
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/llie/LOLv1/Test/input"
        targets = self.args.data_file_dir + "/llie/LOLv1/Test/target"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total LLIE testing pairs : {}".format(len(self.hr)))
            
    ####################################################################################################
    ## DERAINING DATASET
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/deraining/Rain100L/rainy"
        targets = self.args.data_file_dir + "/deraining/Rain100L/gt"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        print("Total Derain testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## DEHAZING DATASET
    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/dehazing/SOTS/outdoor/hazy"
        targets = self.args.data_file_dir + "/dehazing/SOTS/outdoor/gt"

        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        
        self.hr = []
        for sample in self.lr:
            hazy_name = sample["img"]
            clean_name = self._get_nonhazy_name(hazy_name)
            self.hr.append({"img" : clean_name, "de_type":id})
        print("Total Dehazing testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## DENOISING DATASET
    def _init_denoise(self, id):
        inputs = self.args.data_file_dir + "/denoising/cBSD68/original_png"
        
        clean = [x for x in sorted(glob.glob(inputs + "/*.png"))]
        
        self.lr = [{"img" : x, "de_type":id} for x in clean]
        self.hr = [{"img" : x, "de_type":id} for x in clean]
        print("Total Denoise testing pairs : {}".format(len(self.lr)))