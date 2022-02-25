import torch


def load_model(load_type, model_file, device):
    if load_type == 'jit':
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