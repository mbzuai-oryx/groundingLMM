from models.grit_src.image_dense_captions import image_caption_dict

class DenseCaptioning():
    def __init__(self, device):
        self.device = device


    def initialize_model(self):
        pass

    def image_dense_caption_debug(self, image_src):
        dense_caption = """
        1. the broccoli is green, [0, 0, 333, 325]; 
        2. a piece of broccoli, [0, 147, 143, 324]; 
        3. silver fork on plate, [4, 547, 252, 612];
        """
        return dense_caption
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_dict(image_src, self.device)
        return dense_caption
    