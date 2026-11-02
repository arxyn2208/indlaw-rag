# Source and licence

Six Supreme Court Reports (2019) volume-10 judgments, English text extracted from PDFs.
Dataset: "Indian Supreme Court Judgments", Registry of Open Data on AWS,
https://registry.opendata.aws/indian-supreme-court-judgments/ (managed by Dattam Labs).
Licence stated on the registry page: CC-BY-4.0. Attribution: Dattam Labs, accessed 2026-11-03 from the registry URL above.
Original files: s3://indian-supreme-court-judgments/data/pdf/year=2019/english/<name>_EN.pdf. The registry says the judgments were downloaded from the eCourts website.
Changes: text extracted with pdftotext (layout mode); a "CASE TITLE:" line was added. Margin letters, headers and OCR/extraction errors remain.
Not legal advice. The registry licence is the compiler's grant; this repo has not independently verified rights in the underlying court text.

Dataset README (https://github.com/vanga/indian-supreme-court-judgments/blob/main/README.md) also states CC-BY-4.0: "free to use, share, and adapt ... as long as you provide appropriate attribution". Nothing there contradicts the registry page.

## Per-file provenance
Each PDF was fetched from https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/data/pdf/year%3D2019/english/<file> on 2026-11-03.

| File | Bytes | SHA-256 |
|---|---|---|
| 2019_10_1006_1022_EN.pdf | 176973 | 022591c18ef07b8ebf7bfcb306108bbcf02b35da86c4bec631aac44f09c56f76 |
| 2019_10_1023_1040_EN.pdf | 144891 | b8f4b357018fcdce086269805b1508483f0f940a0239b05d2d362145ad698c5a |
| 2019_10_1041_1045_EN.pdf | 139712 | 18f735a4418937e26eb9c0d88e306ee82ace4b44f087b8dad3872a953221e02a |
| 2019_10_1046_1050_EN.pdf | 113644 | e4a2308cbc4fc827a21291d4205b370b6c6add0b1091e544d508070577f3a6f4 |
| 2019_10_1051_1097_EN.pdf | 254709 | 109b64041914339d5c00076b40de7e0f0d8c902ffa0048d3a71dfea1f26c9849 |
| 2019_10_1098_1110_EN.pdf | 162923 | b1c326a66dae83cf454078880a5bca3ab49ffe8a7f6ba3c6c5def9f92e2fc8ff |

Each .txt in this folder has the same stem as its PDF without _EN.

## Paragraph labels
"para N" labels are NOT read from a verified structure. The indlaw chunker splits the extracted text on numbers it detects at line starts, so a label can be wrong, and long paragraphs are cut into "para N(k)" pieces. Treat them as pointers into this text, not as official paragraph numbers of the judgment.
