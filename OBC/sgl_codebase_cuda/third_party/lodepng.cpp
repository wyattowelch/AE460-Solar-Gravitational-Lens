// This project intentionally avoids bundling a large PNG library.
// We output PPM/PGM (portable pixmap/graymap) directly from C++.
// If you need PNGs, run scripts/convert_to_png.py (uses Pillow) or ImageMagick.
// Kept as a stub to preserve include paths used in earlier prototypes.
#include "lodepng.h"
namespace lodepng {
unsigned decode(std::vector<unsigned char>&, unsigned&, unsigned&, const std::string&) { return 1; }
unsigned encode(const std::string&, const std::vector<unsigned char>&, unsigned, unsigned) { return 1; }
const char* error_text(unsigned) { return "PNG support not built-in (use PPM/PGM + conversion script)."; }
}
