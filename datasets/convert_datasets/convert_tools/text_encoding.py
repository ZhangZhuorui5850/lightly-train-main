"""Robust text decoding and in-place UTF-8 normalization for dataset files."""

from __future__ import annotations

import codecs
import os
import stat
import tempfile
import unicodedata
from pathlib import Path


_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)
_COMMON_ENCODINGS = ("utf-8", "gb18030", "big5", "cp1252", "latin-1")
_COMMON_CJK = set(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极车辆测试集"
)


def _detector_encodings(data: bytes) -> list[str]:
    """Ask optional detectors for a hint without making them a dependency."""
    encodings: list[str] = []
    try:
        from charset_normalizer import from_bytes

        match = from_bytes(data).best()
        if match is not None and match.encoding:
            encodings.append(match.encoding)
    except Exception:  # optional detector failures must not block the fallback chain
        pass
    try:
        import chardet

        detected = chardet.detect(data).get("encoding")
        if detected:
            encodings.append(detected)
    except Exception:  # optional detector failures must not block the fallback chain
        pass
    allowed = {
        "utf-8", "utf8", "gb18030", "gbk", "gb2312", "big5", "cp950",
        "cp1252", "windows-1252", "latin-1", "iso-8859-1",
    }
    return [
        encoding
        for encoding in encodings
        if encoding.casefold().replace("_", "-") in allowed
    ]


def _decode(data: bytes, path: Path) -> tuple[str, str]:
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError as exc:
                raise ValueError(f"文本编码损坏: {path} ({encoding}): {exc}") from exc

    # UTF-8 is the repository's canonical encoding.  Many UTF-8 byte strings
    # are also technically decodable as GB18030, so comparing both decoded
    # strings with a language heuristic can corrupt valid Chinese text.
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    candidates: list[str] = []
    ordered_encodings = (
        "gb18030",
        *_detector_encodings(data),
        *_COMMON_ENCODINGS[1:],
    )
    for encoding in ordered_encodings:
        normalized = encoding.casefold().replace("_", "-")
        if normalized not in {item.casefold() for item in candidates}:
            candidates.append(encoding)
    errors: list[str] = []
    decoded: list[tuple[str, str]] = []
    for encoding in candidates:
        try:
            decoded.append((data.decode(encoding), encoding))
        except (LookupError, UnicodeDecodeError) as exc:
            errors.append(f"{encoding}: {exc}")
    if decoded:
        text, encoding = max(decoded, key=lambda item: _text_quality(item[0]))
        return text, encoding
    raise ValueError(f"无法识别文本编码: {path}; {'; '.join(errors)}")


def _text_quality(text: str) -> tuple[int, int, int, int, int]:
    """Rank plausible legacy decodings using script and character quality."""
    cjk = sum(
        0x3400 <= ord(char) <= 0x9FFF or 0xF900 <= ord(char) <= 0xFAFF
        for char in text
    )
    common = sum(char in _COMMON_CJK for char in text)
    private_use = sum(0xE000 <= ord(char) <= 0xF8FF for char in text)
    controls = sum(
        (ord(char) < 32 and char not in "\r\n\t") or ord(char) == 0x7F
        for char in text
    )
    letters = sum(unicodedata.category(char).startswith(("L", "N")) for char in text)
    natural_separators = sum(
        char.isspace() or unicodedata.category(char).startswith("P") for char in text
    )
    return (
        -(private_use + controls),
        common * 8 + letters,
        natural_separators,
        cjk,
        -len(text),
    )


def _write_utf8_atomically(path: Path, text: str) -> None:
    """Rewrite a text file in UTF-8 while retaining permissions and newlines."""
    path = path.expanduser().resolve()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".utf8.tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, original_mode)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def read_text_auto(
    path: Path,
    *,
    rewrite: bool = False,
    encoding: str | None = None,
) -> str:
    """Read text using detected encoding and optionally normalize it to UTF-8.

    Reading is non-mutating by default.  Pass ``rewrite=True`` only from an
    explicit normalization operation; ordinary parsing must never rewrite the
    user's source dataset as a side effect.
    """
    path = path.expanduser().resolve()
    data = path.read_bytes()
    if encoding is None:
        text, detected_encoding = _decode(data, path)
    else:
        try:
            text = data.decode(encoding)
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError(f"文本编码读取失败: {path} ({encoding}): {exc}") from exc
        detected_encoding = encoding
    utf8_data = text.encode("utf-8")
    if rewrite and data != utf8_data:
        _write_utf8_atomically(path, text)
        try:
            from .progress import write as progress_write
        except ImportError:
            from progress import write as progress_write  # type: ignore[no-redef]
        progress_write(f"[编码转换] {path} ({detected_encoding} -> UTF-8)")
    return text


def write_text_utf8(path: Path, text: str) -> None:
    """Write normalized UTF-8 text with exact newline preservation."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
