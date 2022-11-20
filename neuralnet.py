def eval_model_jit(model, data, device):
    import torch

    data = {
        'board_size': torch.tensor(data['board_size'], dtype=torch.int8),
        'board_input': torch.from_numpy(data['board_input']),
        'stm_input': torch.FloatTensor([data['stm_input']]),
    }

    # data placement & add batch dimension
    for k in data:
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

    return value.cpu().numpy(), policy.cpu().numpy()


def eval_model_onnx(session, data, used_input_names):
    import numpy as np

    data = {
        'board_size': data['board_size'],
        'board_input': data['board_input'],
        'stm_input': np.array([data['stm_input']], dtype=np.float32),
    }

    # filter out unused input otherwise onnx will not work
    filtered_data = {}
    for k in used_input_names:
        filtered_data[k] = data[k]
    data = filtered_data

    # add batch dimension
    for k in data:
        data[k] = np.expand_dims(data[k], 0)

    # run onnx inference
    value, policy = session.run(["value", "policy"], data)

    # remove batch dimension
    value = np.squeeze(value, 0)
    policy = np.squeeze(policy, 0)

    # apply activation function
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    policy_shape = policy.shape
    value = softmax(value)
    policy = softmax(policy.flatten()).reshape(policy_shape)

    return value, policy


def load_model(load_type, model_file, device):
    if load_type == 'jit':
        import torch

        model = torch.jit.load(model_file, map_location=device)
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(model)

        return lambda data: eval_model_jit(model, data, device)
    elif load_type == 'onnx':
        import onnxruntime as ort

        providers = []
        for d in device.split(','):
            if d == 'cpu':
                providers.append('CPUExecutionProvider')
            elif d == 'cuda':
                providers.append('CUDAExecutionProvider')
            elif d == "tensorrt":
                providers.append('TensorrtExecutionProvider')
            else:
                raise RuntimeError(f'Unknown device: {d}')

        session = ort.InferenceSession(model_file, providers=providers)
        used_input_names = [n.name for n in session.get_inputs()]

        return lambda data: eval_model_onnx(session, data, used_input_names)
    else:
        assert 0, f"Unsupported load: {load_type}"