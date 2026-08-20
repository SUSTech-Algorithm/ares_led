from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml

Color = Tuple[int, int, int]


class FixedColorProtocol:
    """整数命令与颜色符号对之间的双向映射。"""

    def __init__(
            self,
            config_path: Optional[str] = None,
            colors_path: Optional[str] = None) -> None:
        if config_path:
            path = Path(config_path)
        else:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = Path(get_package_share_directory('rgb_comm_protocol')) / 'config/rgb_protocol.yaml'
            except Exception:
                path = Path(__file__).resolve().parents[1] / 'config/rgb_protocol.yaml'

        with path.open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)

        self.colors: Dict[str, Color] = {}
        if colors_path:
            with Path(colors_path).open('r', encoding='utf-8') as stream:
                color_data = yaml.safe_load(stream) or {}
            self.colors = {
                symbol: tuple(int(channel) for channel in rgb)
                for symbol, rgb in color_data['colors'].items()
            }
        self.commands: Dict[int, Tuple[str, ...]] = {
            int(command): self._normalize_code(code)
            for command, code in data['commands'].items()
        }
        if any(len(code) != 2 for code in self.commands.values()):
            raise ValueError('every protocol code must contain exactly two symbols')
        if any(code[0] == code[1] for code in self.commands.values()):
            raise ValueError('the two symbols in each protocol code must differ')
        if len(set(self.commands.values())) != len(self.commands):
            raise ValueError('protocol codes must be unique')
        if self.colors:
            unknown = {
                symbol
                for code in self.commands.values()
                for symbol in code
                if symbol not in self.colors
            }
            if unknown:
                raise ValueError(f'unknown color symbols: {sorted(unknown)}')
        self._decode: Dict[Tuple[str, ...], int] = {}
        for command, code in self.commands.items():
            for accepted in {code, tuple(reversed(code))}:
                owner = self._decode.get(accepted)
                if owner is not None and owner != command:
                    raise ValueError(
                        f'protocol codes {code} and {accepted} conflict under reversal')
                self._decode[accepted] = command
        self.accepted_codes = frozenset(self._decode)

    @staticmethod
    def _normalize_code(code) -> Tuple[str, ...]:
        if isinstance(code, str):
            values = [item.strip() for item in code.replace(',', ' ').split()]
        else:
            values = list(code)
        return tuple(str(item).upper() for item in values)

    def encode_symbols(self, command_id: int) -> Optional[Tuple[str, ...]]:
        return self.commands.get(int(command_id))

    def encode_rgb(self, command_id: int) -> Optional[Tuple[Color, ...]]:
        if not self.colors:
            raise RuntimeError('RGB values are endpoint-specific; provide colors_path')
        symbols = self.encode_symbols(command_id)
        if symbols is None:
            return None
        return tuple(self.colors[symbol] for symbol in symbols)  # type: ignore[return-value]

    def decode(self, symbols: Iterable[str]) -> Optional[int]:
        return self._decode.get(tuple(str(symbol).upper() for symbol in symbols))
