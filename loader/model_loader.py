import settings
import torch
import torchvision
import torch.nn as nn
import functools


def noop(*args, **kwargs):
    pass


def Conv4(*args, num_classes=200, **kwargs):
    # Exactly the same as few-shot literature
    cn = ConvNet(4, num_classes, *args, **kwargs, pool_size=2, padding=1)
    return cn


class ConvNet(nn.Module):
    def __init__(self, depth, num_classes, num_channels=32, grayscale=False, pretrained=False, **kwargs):
        if pretrained:
            raise NotImplementedError
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            if i == 0:
                if grayscale:
                    indim = 1
                else:
                    indim = 3
            else:
                indim = num_channels
            outdim = num_channels
            B = ConvBlock(indim, outdim, pool=(i < 4), **kwargs)
            trunk.append(B)

        #  trunk.append(nn.Flatten())

        self.num_classes = num_classes
        self.trunk = nn.Sequential(*trunk)
        self.output_dim = num_channels
        self.fc = nn.Linear(self.output_dim, self.num_classes)

    def forward(self, x):
        x_enc = self.trunk(x)
        # Do average over patches
        x_enc = x_enc.mean(2).mean(2)
        preds = self.fc(x_enc)
        return preds

    def reset_parameters(self):
        # Skip flatten
        for cb in self.trunk[:-1]:
            cb.reset_parameters()
        self.linear.reset_parameters()


class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True, pool_size=2, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        # self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        #  self.parametrized_layers = [self.C, self.BN, self.relu]
        self.parametrized_layers = [self.C, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(pool_size)
            self.parametrized_layers.append(self.pool)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        self.C.reset_parameters()
        # self.BN.reset_parameters()


def loadmodel(
    hook_fn,
    feature_names=settings.FEATURE_NAMES,
    hook_modules=None,
    pretrained_override=None,
    num_channels=32,  # Applicable to conv4 only
):
    device = torch.device("cuda" if settings.GPU else "cpu")
    if settings.MODEL == "conv4":
        model_fn = functools.partial(Conv4, num_channels=num_channels)
    else:
        model_fn = torchvision.models.__dict__[settings.MODEL]

    if settings.MODEL_FILE is None:
        if settings.MODEL == "conv4":
            raise NotImplementedError("No pretrained conv4")
        model = model_fn(
            pretrained=pretrained_override if pretrained_override is not None else True
        )
    elif settings.MODEL_FILE == "<UNTRAINED>":
        model = model_fn(
            pretrained=pretrained_override if pretrained_override is not None else False
        )
    else:
        checkpoint = torch.load(settings.MODEL_FILE, map_location=device)
        if (
            type(checkpoint).__name__ == "OrderedDict"
            or type(checkpoint).__name__ == "dict"
        ):
            model = model_fn(num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {
                    str.replace(k, "module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    if hook_fn is not None:
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrive the module
                hook_model = model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            if hook_modules is not None:
                hook_modules.append(hook_model)
            hook_model.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
