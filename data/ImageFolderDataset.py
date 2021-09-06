import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    print("can't find acc image loader")
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, dir, transform=None):
        image_list = []
        for target in sorted(os.listdir(dir)):
            path = os.path.join(dir, target)
            image_list.append(path)

        self.image_list = image_list
        self.loader = default_loader
        self.transform = transform
        self.len = len(image_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.image_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.len


class NewMyImageFolder(data.Dataset):
    def __init__(self, dir, transform=None):
        super(NewMyImageFolder, self).__init__()
        self.image_paths = []
        image_dirs = os.listdir(dir)
        self.loader = default_loader
        for dir_name in image_dirs:
            base_name = os.path.join(dir, dir_name)
            file_names = os.listdir(base_name)
            for file_name in file_names:
                self.image_paths.append(os.path.join(base_name, file_name))
        self.len = len(self.image_paths)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img