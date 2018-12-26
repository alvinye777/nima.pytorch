import torch
from torchvision.datasets.folder import default_loader

from decouple import config

from nima.model import NIMA
from nima.common import Transform, get_mean_score, get_std_score
from nima.common import download_file
from nima.inference.utils import format_output

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class InferenceModel:
    @classmethod
    def create_model(cls):
        path_to_model = download_file(config('MODEL_URL'), config('MODEL_PATH'))
        return cls(path_to_model)

    def __init__(self, path_to_model):
        self.transform = Transform().val_transform
        self.model = NIMA(pretrained_base_model=False)
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image):
        image = image.convert('RGB')
        return self.predict(image)

    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        image = torch.autograd.Variable(image, volatile=True)
        prob = self.model(image).data.cpu().numpy()[0]

        mean_score = get_mean_score(prob)
        std_score = get_std_score(prob)

        return format_output(mean_score, std_score, prob)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.autograd import Variable
    # preprocess
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # NIMA
    nima_model = NIMA(pretrained_base_model=False)
    # load pretrain
    path_to_model = download_file(config('MODEL_URL'), config('MODEL_PATH'))
    state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
    nima_model.load_state_dict(state_dict)
    # set model as evaluation
    nima_model.eval()
    # Input to the model
    batch_size = 1
    input_var = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)
    # Export the model
    torch_out = torch.onnx.export(nima_model, input_var, "nima_model.onnx", export_params=True)
