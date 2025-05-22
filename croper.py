import PIL
import numpy as np

from PIL import Image

class Croper:
    def __init__(
        self,
        input_image: PIL.Image,
        target_mask: np.ndarray,
        mask_size: int = 256,
        mask_expansion: int = 20,
    ):
        self.input_image = input_image
        self.target_mask = target_mask
        self.mask_size = mask_size
        self.mask_expansion = mask_expansion
    
    def corp_mask_image(self):
        target_mask = self.target_mask
        input_image = self.input_image
        mask_expansion = self.mask_expansion
        original_width, original_height = input_image.size
        mask_indices = np.where(target_mask)
        start_y = np.min(mask_indices[0])
        end_y = np.max(mask_indices[0])
        start_x = np.min(mask_indices[1])
        end_x = np.max(mask_indices[1])
        mask_height = end_y - start_y
        mask_width = end_x - start_x
        # choose the max side length
        max_side_length = max(mask_height, mask_width)
        # expand the mask area
        height_diff = (max_side_length - mask_height) // 2
        width_diff = (max_side_length - mask_width) // 2
        start_y = start_y - mask_expansion - height_diff
        if start_y < 0:
            start_y = 0
        end_y = end_y + mask_expansion + height_diff
        if end_y > original_height:
            end_y = original_height
        start_x = start_x - mask_expansion - width_diff
        if start_x < 0:
            start_x = 0
        end_x = end_x + mask_expansion + width_diff
        if end_x > original_width:
            end_x = original_width
        expanded_height = end_y - start_y
        expanded_width = end_x - start_x
        expanded_max_side_length = max(expanded_height, expanded_width)
        # calculate the crop area
        crop_mask = target_mask[start_y:end_y, start_x:end_x]
        crop_mask_start_y = (expanded_max_side_length - expanded_height) // 2
        crop_mask_end_y = crop_mask_start_y + expanded_height
        crop_mask_start_x = (expanded_max_side_length - expanded_width) // 2
        crop_mask_end_x = crop_mask_start_x + expanded_width
        # create a square mask
        square_mask = np.zeros((expanded_max_side_length, expanded_max_side_length), dtype=target_mask.dtype)
        square_mask[crop_mask_start_y:crop_mask_end_y, crop_mask_start_x:crop_mask_end_x] = crop_mask
        square_mask_image = Image.fromarray((square_mask * 255).astype(np.uint8))

        crop_image = input_image.crop((start_x, start_y, end_x, end_y))
        square_image = Image.new("RGB", (expanded_max_side_length, expanded_max_side_length))
        square_image.paste(crop_image, (crop_mask_start_x, crop_mask_start_y))

        self.origin_start_x = start_x
        self.origin_start_y = start_y
        self.origin_end_x = end_x
        self.origin_end_y = end_y

        self.square_start_x = crop_mask_start_x
        self.square_start_y = crop_mask_start_y
        self.square_end_x = crop_mask_end_x
        self.square_end_y = crop_mask_end_y

        self.square_length = expanded_max_side_length
        self.square_mask_image = square_mask_image
        self.square_image = square_image
        self.corp_mask = crop_mask

        mask_size = self.mask_size
        self.resized_square_mask_image = square_mask_image.resize((mask_size, mask_size))
        self.resized_square_image = square_image.resize((mask_size, mask_size))

        return self.resized_square_mask_image
    
    def restore_result(self, generated_image):
        square_length = self.square_length
        generated_image = generated_image.resize((square_length, square_length))
        square_mask_image = self.square_mask_image
        cropped_generated_image = generated_image.crop((self.square_start_x, self.square_start_y, self.square_end_x, self.square_end_y))
        cropped_square_mask_image = square_mask_image.crop((self.square_start_x, self.square_start_y, self.square_end_x, self.square_end_y))

        restored_image = self.input_image.copy()
        restored_image.paste(cropped_generated_image, (self.origin_start_x, self.origin_start_y), cropped_square_mask_image)
        
        return restored_image
    
    def restore_result_v2(self, generated_image):
        square_length = self.square_length
        generated_image = generated_image.resize((square_length, square_length))
        cropped_generated_image = generated_image.crop((self.square_start_x, self.square_start_y, self.square_end_x, self.square_end_y))

        restored_image = self.input_image.copy()
        restored_image.paste(cropped_generated_image, (self.origin_start_x, self.origin_start_y))
        
        return restored_image
        