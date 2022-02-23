import torch
import torch.nn as nn


def build_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'none':
        return None
    else:
        assert 0, f"Unsupported activation: {activation}"


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu', bias=True):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # initialize activation
        self.activation = build_activation_layer(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 bias=True,
                 dilation=(1, ),
                 groups=1,
                 activation_first=False,
                 use_spectral_norm=False):
        super(Conv2dBlock, self).__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # initialize activation
        self.activation = build_activation_layer(activation)

        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            ks,
            st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self,
                 dim,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 dim_hidden=None):
        super(ResBlock, self).__init__()
        dim_hidden = dim_hidden or dim

        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(dim, dim_hidden, ks, st, padding, norm, activation, pad_type),
            Conv2dBlock(dim_hidden, dim, ks, st, padding, norm, 'none', pad_type),
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        if self.activation:
            out = self.activation(out)
        return out


def build_input_plane(input_type):
    if input_type == 'basic':
        return BasicInputPlane(with_stm=True)
    elif input_type == 'basic-nostm':
        return BasicInputPlane(with_stm=False)
    elif input_type == 'raw':
        return lambda x: x  # identity transform
    else:
        assert 0, f"Unsupported input: {input_type}"


class BasicInputPlane(nn.Module):
    def __init__(self, with_stm=True):
        super().__init__()
        self.with_stm = with_stm

    def forward(self, data):
        board_input = data['board_input'].float()
        stm_input = data['stm_input'].float()

        if self.with_stm:
            B, C, H, W = board_input.shape
            stm_input = stm_input.reshape(B, 1, 1, 1).expand(B, 1, H, W)
            input_plane = torch.cat([board_input, stm_input], dim=1)
        else:
            input_plane = board_input

        return input_plane

    @property
    def dim_plane(self):
        return 2 + self.with_stm


def build_head(head_type, dim_feature):
    if '-nodraw' in head_type:
        dim_value = 1
        head_type = head_type.replace('-nodraw', '')
    else:
        dim_value = 3

    if head_type == 'v0':
        return OutputHeadV0(dim_feature, dim_value)
    elif head_type == 'v1' or head_type.starts_with('v1-'):
        scale = int(head_type[3:] or 1)
        return OutputHeadV1(dim_feature, dim_feature * scale, dim_value)
    else:
        assert 0, f"Unsupported head: {head_type}"


class OutputHeadV0(nn.Module):
    def __init__(self, dim_feature, dim_value=3):
        super().__init__()
        self.value_head = LinearBlock(dim_feature, dim_value, activation='none')
        self.policy_head = Conv2dBlock(dim_feature, 1, ks=1, st=1, activation='none')

    def forward(self, feature):
        # value head
        value = torch.mean(feature, dim=(2, 3))
        value = self.value_head(value)

        # policy head
        policy = self.policy_head(feature)
        policy = torch.squeeze(policy, dim=1)

        return value, policy


class OutputHeadV1(OutputHeadV0):
    def __init__(self, dim_feature, dim_middle, dim_value=3):
        super().__init__(dim_middle, dim_value)
        self.conv = Conv2dBlock(dim_feature, dim_middle, ks=3, st=1, padding=1)

    def forward(self, feature):
        feature = self.conv(feature)
        return super().forward(feature)


class ResNet(nn.Module):
    def __init__(self, num_blocks, dim_feature, head_type='v0', input_type='basic'):
        super().__init__()
        self.model_size = (num_blocks, dim_feature)
        self.head_type = head_type
        self.input_type = input_type

        self.input_plane = build_input_plane(input_type)
        self.conv_input = Conv2dBlock(self.input_plane.dim_plane,
                                      dim_feature,
                                      ks=3,
                                      st=1,
                                      padding=1,
                                      activation='lrelu')
        self.conv_trunk = nn.Sequential(
            *[ResBlock(dim_feature, norm='bn') for i in range(num_blocks)])
        self.output_head = build_head(head_type, dim_feature)

    def forward(self, data):
        input_plane = self.input_plane(data)
        feature = self.conv_input(input_plane)
        feature = self.conv_trunk(feature)
        return self.output_head(feature)

    @property
    def name(self):
        b, f = self.model_size
        return f"resnet_{self.input_type}_{b}b{f}f{self.head_type}"


MODELS = {
    'resnet': ResNet,
}


def load_model(load_type, model_file, model_type, device, **kwargs):
    if load_type == 'pytorch':
        # build model
        model = MODELS[model_type](**kwargs).to(device)
        model.eval()
        # load model file
        state_dicts = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dicts['model'])
    elif load_type == 'jit':
        model = torch.jit.load(model_file, map_location=device)
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(model)
    else:
        assert 0, f"Unsupported load: {load_type}"
    return model


def eval_model(model, data, device):
    # data placement & add batch dimension
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = torch.unsqueeze(data[k].to(device), dim=0)

    with torch.no_grad():
        value, policy = model(data)

    # remove batch dimension
    value = value.squeeze(0)
    policy = policy.squeeze(0)

    # apply activation function
    policy_shape = policy.shape
    value = torch.softmax(value, dim=0)
    policy = torch.softmax(policy.flatten(), dim=0).reshape(policy_shape)

    return value, policy