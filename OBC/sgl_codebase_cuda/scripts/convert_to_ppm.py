#!/usr/bin/env python3
import sys
from PIL import Image

def main():
    if len(sys.argv) < 3:
        print("Usage: convert_to_ppm.py input_image output.ppm")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    img = Image.open(inp).convert("RGB")
    img.save(outp, format="PPM")
    print(f"Wrote {outp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
