"""Derive Chrome SODA component IDs from Chromium's public constants.h.

This does not download Google binaries or bypass access controls.
"""
import re
import urllib.request

URL = "https://raw.githubusercontent.com/chromium/chromium/main/components/soda/constants.h"
text = urllib.request.urlopen(URL, timeout=20).read().decode()
for locale in ("ja-JP", "vi-VN"):
    pattern = r'\{LanguageCode::[A-Za-z]+,\s*"' + re.escape(locale) + r'"[\s\S]*?\{([^}]+)\}\}'
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Could not find {locale} in Chromium constants.h")
    b = bytes(int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", match.group(1)))
    assert len(b) == 32, (locale, len(b))
    extension_id = "".join(chr(ord("a") + (v >> 4)) + chr(ord("a") + (v & 15)) for v in b[:16])
    print(locale, extension_id, b.hex())
