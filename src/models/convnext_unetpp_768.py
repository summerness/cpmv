from models.convnext_unetpp_512 import ConvNeXtUNetPP512


class ConvNeXtUNetPP768(ConvNeXtUNetPP512):
    """ConvNeXt-Tiny UNet++ model tailored for 768x768 inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
