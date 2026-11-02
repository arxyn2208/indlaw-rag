# Source and licence

Forty Supreme Court Reports judgments (English text extracted from PDFs; years 2016 to 2022, 2019 over-represented because the first six files came from it).
Selection: not curated for importance or topic. Files of 70-350 KB were taken at even steps from each year's bucket listing, plus six consecutive volume-10 2019 files. No judgment was chosen for its outcome.
Dataset: "Indian Supreme Court Judgments", Registry of Open Data on AWS, https://registry.opendata.aws/indian-supreme-court-judgments/ (managed by Dattam Labs).
Licence stated on the registry page: CC-BY-4.0. The dataset README (https://github.com/vanga/indian-supreme-court-judgments/blob/main/README.md) says the same: "free to use, share, and adapt ... as long as you provide appropriate attribution".
Attribution: Dattam Labs, Indian Supreme Court Judgments, accessed 2026-11-03 from the registry URL above. The registry says the judgments were downloaded from the eCourts website.
Changes: text extracted with pdftotext (layout mode); a "CASE TITLE:" line was added by scripts/prepare_sc.py from the party names near the first "v." line. Some titles are cut or garbled (for example "ANDANR v. UNION OF INDIA AND ORS"). Margin letters, headers and extraction errors remain.
Not legal advice. The registry licence is the compiler's grant; this repo has not independently verified rights in the underlying court text.

## Per-file provenance
Each PDF was fetched on 2026-11-03 from https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/data/pdf/year%3D<year>/english/<file>. The .txt in this folder has the same stem as its PDF without _EN.

| File | Year | Bytes | SHA-256 |
|---|---|---|---|
| 2019_10_1006_1022_EN.pdf | 2019 | 176973 | 022591c18ef07b8ebf7bfcb306108bbcf02b35da86c4bec631aac44f09c56f76 |
| 2019_10_1023_1040_EN.pdf | 2019 | 144891 | b8f4b357018fcdce086269805b1508483f0f940a0239b05d2d362145ad698c5a |
| 2019_10_1041_1045_EN.pdf | 2019 | 139712 | 18f735a4418937e26eb9c0d88e306ee82ace4b44f087b8dad3872a953221e02a |
| 2019_10_1046_1050_EN.pdf | 2019 | 113644 | e4a2308cbc4fc827a21291d4205b370b6c6add0b1091e544d508070577f3a6f4 |
| 2019_10_1051_1097_EN.pdf | 2019 | 254709 | 109b64041914339d5c00076b40de7e0f0d8c902ffa0048d3a71dfea1f26c9849 |
| 2019_10_1098_1110_EN.pdf | 2019 | 162923 | b1c326a66dae83cf454078880a5bca3ab49ffe8a7f6ba3c6c5def9f92e2fc8ff |
| 2016_11_210_218_EN.pdf | 2016 | 304646 | 81fbe4dc1e942a0911792e433a71420f18c53475b409c649715f53924e1df91f |
| 2016_2_144_149_EN.pdf | 2016 | 159868 | ffddc44c265eb2ce9d94fc2d62a6cf67d34b4f147dc0005739113bcb5c55111b |
| 2016_2_914_919_EN.pdf | 2016 | 157640 | 1576bbc3ad2127bf86274f83ccfdbf2c421f48fc0a09caf5ee928ee4021ad4ff |
| 2016_4_627_634_EN.pdf | 2016 | 251746 | 1b23e7a499f46b57afa7ee77903d9ec695b87370f394d32ce487ce95561e414b |
| 2017_10_1000_1005_EN.pdf | 2017 | 230797 | 3c6628bb06e714176fd31d0417a9d8e47e2316acb3839e8e0dae7abd59809735 |
| 2017_14_479_487_EN.pdf | 2017 | 301988 | 0e9f8513bb52bfd300ac4ae366e2a3eb4aef9d3d6428352c90000d175fc8ee53 |
| 2017_2_889_895_EN.pdf | 2017 | 209483 | 2ae74fdad7b796ddabd967b91edb9dfab43a0ba5da283e9a13a171a1a6db3031 |
| 2017_6_462_471_EN.pdf | 2017 | 308311 | 088757a3577c4e17ace5ef231b2845f940bc0c6d1f5a86223de7c7dcc6c7ab3d |
| 2018_10_100_122_EN.pdf | 2018 | 253437 | 91aac90e78efea753a88799e0735c4fdba8a0ea7698b727ab0eb885e0fc21c22 |
| 2018_13_1312_1318_EN.pdf | 2018 | 130356 | ed0f379e1048b70166a704e95c39d0ff9fee926dbd1c8140d5c1b269a9e7afd7 |
| 2018_1_626_635_EN.pdf | 2018 | 187869 | 1fa2a67c9224f9a78b3c91786c06884bd2bc84bd3314c579a982a4618a822aff |
| 2018_4_347_361_EN.pdf | 2018 | 210685 | 62480a02a2c256464c0176af732e16f84fdd95077b585d81e3594a5561d193da |
| 2019_10_1111_1120_EN.pdf | 2019 | 153040 | 6067f4716589de00145a0f79c699f39f0c399c3a0d1b795dbb24f644cf5e33ed |
| 2019_11_30_39_EN.pdf | 2019 | 269756 | ec2066de997ea77b271a9947a5a969d0d9e026a5f3de3ec83ff0e1d9dc82817a |
| 2019_13_1028_1043_EN.pdf | 2019 | 210095 | 816dcac18d02b581383b52853d734885fbe7815ef517ce759ebf6bc7c6f1892e |
| 2019_14_818_828_EN.pdf | 2019 | 188067 | 42ffa115d1ddbb5f957b631be0acca9023787d37248a327e9202a230d74580b9 |
| 2019_17_986_994_EN.pdf | 2019 | 145068 | 2770243bdade9087f98d29f5973c432984923b5772c57a39cb689d9f8ae67caf |
| 2019_2_1025_1039_EN.pdf | 2019 | 198786 | f2ca581a309bf67af9bc1d141fea470d017f9228e40b7aab73f601ebb296912d |
| 2019_3_357_366_EN.pdf | 2019 | 155920 | 9170fd587187824275824d66963433fd18dc0ffbd5d42c9e0b8c7e80cf9bed6e |
| 2019_4_868_874_EN.pdf | 2019 | 118486 | 747932d068714179b1ae9489457006d80ad246f2aed8bce1d547ebde3528019d |
| 2019_6_1158_1171_EN.pdf | 2019 | 304312 | afb2a9878b54663c8e0530f804c4e687dc2cc97477eacf60beeebc28c933406f |
| 2019_7_361_369_EN.pdf | 2019 | 245763 | 7674ffddd9b2dee2040f735ace8c7cd2e5dfca3a7eb0eefcc87efe38696d9168 |
| 2020_10_1132_1150_EN.pdf | 2020 | 342757 | 1cd1ddc5ca52bfd62d8ffc8935b42a54958a58fda0df779749961efafe2e5135 |
| 2020_1_69_75_EN.pdf | 2020 | 142988 | f5bb53d84f32bbdb3df6faa41451ec2cf7fc7d57bae2d18bf333f957272b0a6e |
| 2020_3_824_832_EN.pdf | 2020 | 150460 | 20634b4a338abaa45db6585240ea6615905a7653ac7e186f65b1d8f8504f0a5c |
| 2020_5_59_69_EN.pdf | 2020 | 203748 | bfe9d3962dfede312087319481e4506a32b0234c78a33f109f8b28a5fb152398 |
| 2021_10_1009_1021_EN.pdf | 2021 | 348698 | e835b59ac519f54cc938e7b7937ad742ec8b8f707e10209efce2005a9acf4ace |
| 2021_1_701_707_EN.pdf | 2021 | 164738 | 0aa788b9cb6f73e78f85254223a47f17c52fb24d719448719935a9767931b14c |
| 2021_6_208_218_EN.pdf | 2021 | 167752 | 33d34c7f89fad3bceb83a692af1a428c68fc7298fe08367024277d54dc6815ff |
| 2021_8_31_38_EN.pdf | 2021 | 245675 | 50287fe64c9b9e847797a9681b7ac61fc45688a7b870bba8ca60b773042987cb |
| 2022_10_1106_1134_EN.pdf | 2022 | 331326 | 8d34b6d8f6ce89ebb75d3176d175e09bbe38d3f53f4ba2d55b68b9ef6f6cc18f |
| 2022_16_522_542_EN.pdf | 2022 | 275588 | e97d7325fd23f1c4629d011e16f7fd77638b012580aff34c58911c28abace323 |
| 2022_2_120_125_EN.pdf | 2022 | 132841 | bda468318b44a44759b0f756440ef9301c2eae3b3bfd2c18fb5e97f634cbbfe1 |
| 2022_5_613_628_EN.pdf | 2022 | 217619 | 5f6af03c971d110605b82a42fabad446721b8662667be956aa190077c775fc84 |

## Paragraph labels
"para N" labels are NOT read from a verified structure. The indlaw chunker splits the extracted text on numbers it detects at line starts, so a label can be wrong, and long paragraphs are cut into "para N(k)" pieces. Treat them as pointers into this text, not as official paragraph numbers of the judgment.
