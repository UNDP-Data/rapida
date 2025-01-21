import pytest
from typing import List
from click.testing import CliRunner
from cbsurge.cli import cli  # あなたの CLI コマンドが定義されているモジュールをインポート


@pytest.mark.parametrize(
    "commands, texts",
    [
        # test for rapida --help
        (
            ["--help"],
            ["Main CLI for the application."]
        ),
        # test for rapida init --help
        (
            ["init", "--help"],
            ["This command setup rapida command environment by authenticating to Azure."]
        ),
        # test for rapida admin --help
        (
            ["admin", "--help"],
            ["ocha", "osm"]
        ),
        # test for rapida admin ocha --help
        (
            ["admin", "ocha", "--help"],
            ["Fetch admin boundaries from OCHA COD"]
        ),
        # test for rapida admin osm --help
        (
            ["admin", "osm", "--help"],
            ["Fetch admin boundaries from OSM"]
        ),
        # test for rapida population --help
        (
            ["population", "--help"],
            ["download", "run-aggregate", "sync"]
        ),
        # test for rapida population download --help
        (
            ["population", "download", "--help"],
            ["population download"]
        ),
        # test for rapida population run-aggregate --help
        (
            ["population", "run-aggregate", "--help"],
            ["population run-aggregate"]
        ),
        # test for rapida population sync --help
        (
            ["population", "sync", "--help"],
            ["population sync"]
        ),
        # test for rapida stats --help
        (
            ["stats", "--help"],
            ["compute",]
        ),
        # test for rapida stats compute --help
        (
            ["stats", "compute", "--help"],
            ["This command compute zonal statistics with given raster files from a vector",]
        ),
    ]
)
def test_cli(commands: List[str], texts: List[str]) -> None:
    """
    Test CLI command to ensure it is executable.
    Args:
        commands (List[str]): The CLI command and its arguments as a list of strings.
            For example, ["init", "--help"].
        texts (List[str]): A list of strings expected to appear in the command output.
            For example, ["Main CLI for the application."].
    """
    runner = CliRunner()
    result = runner.invoke(cli, commands)
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}"
    for text in texts:
        assert text in result.output, f"{commands.join(" ")} command help not found"