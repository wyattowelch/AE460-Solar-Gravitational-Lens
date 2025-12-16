#!/usr/bin/env python3
import sys
from PIL import Image

def main():
    if len(sys.argv) < 3:
        print("Usage: convert_ppm_to_png.py input.ppm output.png")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    img = Image.open(inp)
    img.save(outp, format="PNG")
    print(f"Wrote {outp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
