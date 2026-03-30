import pytest

from swift.cli.main import cli_main


def test_swift_without_args_prints_help(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['swift'])

    cli_main()

    captured = capsys.readouterr()
    assert 'Usage: swift <command> [args]' in captured.out
    assert 'Available commands:' in captured.out
    assert 'sft' in captured.out
    assert 'infer' in captured.out


def test_swift_with_unknown_command_prints_help(monkeypatch):
    monkeypatch.setattr('sys.argv', ['swift', 'unknown'])

    with pytest.raises(SystemExit, match='Unknown command: unknown'):
        cli_main()
