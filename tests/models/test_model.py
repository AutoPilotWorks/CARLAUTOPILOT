import torch

from carlautopilot.models.route_steering import NUM_COMMANDS, RouteSteeringNet


def batch(n=6, h=32, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, h, w, generator=g), torch.rand(n, generator=g) * 10


def test_output_shape_and_ranges_for_every_command():
    torch.manual_seed(0)
    net = RouteSteeringNet()
    img, spd = batch()
    for c in range(NUM_COMMANDS):
        out = net(img, spd, torch.full((6,), c))
        assert out.shape == (6, 3)
        assert out[:, 0].abs().max() <= 1.0
        assert out[:, 1:].min() >= 0.0 and out[:, 1:].max() <= 1.0


def test_heads_differ_by_command():
    torch.manual_seed(0)
    net = RouteSteeringNet()
    img, spd = batch()
    outs = [net(img, spd, torch.full((6,), c)) for c in range(NUM_COMMANDS)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])


def test_gradients_reach_only_selected_head():
    torch.manual_seed(0)
    net = RouteSteeringNet()
    img, spd = batch()
    net(img, spd, torch.full((6,), 2)).sum().backward()
    for i, head in enumerate(net.heads):
        total = sum(float(p.grad.abs().sum()) for p in head.parameters())
        assert (total > 0) == (i == 2)
    assert sum(float(p.grad.abs().sum()) for p in net.backbone.parameters()) > 0


def test_mixed_commands_in_one_batch():
    torch.manual_seed(0)
    net = RouteSteeringNet()
    img, spd = batch(4)
    cmd = torch.tensor([0, 1, 2, 3])
    mixed = net(img, spd, cmd)
    for i in range(4):
        single = net(img[i : i + 1], spd[i : i + 1], cmd[i : i + 1])
        assert torch.allclose(mixed[i : i + 1], single, atol=1e-6)


def test_handles_other_image_sizes():
    net = RouteSteeringNet()
    out = net(torch.rand(2, 3, 48, 96), torch.rand(2), torch.tensor([0, 1]))
    assert out.shape == (2, 3)
