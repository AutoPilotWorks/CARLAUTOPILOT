import socket

from carlautopilot.cli import check_env


def test_module_status():
    assert check_env.check_module("json")[0] == check_env.OK
    assert check_env.check_module("no_such_module_xyz")[0] == check_env.FAIL
    assert check_env.check_module("no_such_module_xyz", required=False)[0] == check_env.WARN


def test_server_reachable_and_not():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    assert check_env.check_server("127.0.0.1", port)[0] == check_env.OK
    s.close()
    assert check_env.check_server("127.0.0.1", port, timeout=0.5)[0] == check_env.WARN


def test_disk_threshold():
    assert check_env.check_disk(".", need_gib=0.0)[0] == check_env.OK
    assert check_env.check_disk(".", need_gib=1e9)[0] == check_env.WARN


def test_main_exit_code(capsys):
    code = check_env.main([])
    out = capsys.readouterr().out
    assert "python" in out and "carla-server" in out
    assert code in (0, 1)
