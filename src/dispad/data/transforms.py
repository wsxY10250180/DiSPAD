import logging
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class _AugmentationCompose(T.Compose):
    """Compose augmentations with a rotation angle supplied for each sample."""

    def __call__(self, image, rotation_angle=0.0):
        for transform in self.transforms:
            if transform is TF.rotate:
                image = transform(image, rotation_angle)
            else:
                image = transform(image)
        return image


def get_augmentation_transform(aug_list: list, image_res: int):
    """
    Builds a torchvision.transforms.Compose object from a list of augmentation names.
    """
    transforms_list = []
    if not aug_list:
        return _AugmentationCompose([])

    logging.info("Initializing Augmentations")
    for aug_name in aug_list:
        if aug_name == "hflip":
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            logging.info("Added augmentation: RandomHorizontalFlip(p=0.5)")
        elif aug_name == "vflip":
            transforms_list.append(T.RandomVerticalFlip(p=0.5))
            logging.info("Added augmentation: RandomVerticalFlip(p=0.5)")
        elif aug_name == "rotate":
            transforms_list.append(TF.rotate)
            logging.info("Added augmentation: fixed rotation (360 / aug_count degree steps)")
        elif aug_name == "color_jitter":
            transforms_list.append(
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )
            logging.info(
                "Added augmentation: ColorJitter(brightness=0.2, contrast=0.2, ...)"
            )
        elif aug_name == "affine":
            transforms_list.append(
                T.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10)
            )
        else:
            logging.warning(f"Unknown augmentation '{aug_name}' requested. Ignoring.")

    if not transforms_list:
        logging.warning(
            "Augmentation requested but no valid augmentations found in aug_list."
        )
        return _AugmentationCompose([])

    return _AugmentationCompose(transforms_list)
