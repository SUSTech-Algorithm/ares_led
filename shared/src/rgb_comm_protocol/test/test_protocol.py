from rgb_comm_protocol import FixedColorProtocol
import pytest


def test_all_commands_round_trip():
    protocol = FixedColorProtocol()
    assert set(protocol.commands) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert protocol.encode_symbols(0) == ('RED', 'GREEN')
    assert protocol.encode_symbols(1) == ('RED', 'BLUE')
    assert protocol.decode(('RED', 'GREEN')) == 0
    assert protocol.decode(('GREEN', 'RED')) == 0
    for command, symbols in protocol.commands.items():
        assert protocol.decode(symbols) == command
        assert protocol.decode(reversed(symbols)) == command
        assert len(symbols) == 2
        assert symbols[0] != symbols[1]
    assert all(
        len(set(symbols)) > 1
        for symbols in protocol.commands.values()
    )
    equivalence_classes = {
        min(symbols, symbols[::-1])
        for symbols in protocol.commands.values()
    }
    assert len(equivalence_classes) == len(protocol.commands)


def test_unknown_values_are_rejected():
    protocol = FixedColorProtocol()
    assert protocol.encode_symbols(99) is None
    assert protocol.decode(('CYAN', 'PURPLE')) is None


def test_reversed_commands_are_rejected_as_conflicts(tmp_path):
    config = tmp_path / 'protocol.yaml'
    config.write_text(
        'commands:\n'
        '  1: [RED, BLUE]\n'
        '  2: [BLUE, RED]\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='conflict under reversal'):
        FixedColorProtocol(config_path=str(config))
