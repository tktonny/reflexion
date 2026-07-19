# Final Balanced Qwen Omni Benchmark

- English cases: 50
- Chinese cases: 30
- Combined cases: 80
- Note: executed the explicit 50 English + 30 Chinese quota.

## Overall Metrics

- Accuracy: 0.6375
- Precision: 0.6037735849056604
- Recall: 0.8
- Specificity: 0.475
- F1: 0.6881720430107526
- Confusion Matrix: {"true_positive": 32, "true_negative": 19, "false_positive": 21, "false_negative": 8}

## Metrics By Language

### English

- Accuracy: 0.74
- Precision: 0.7307692307692307
- Recall: 0.76
- Specificity: 0.72
- F1: 0.7450980392156863
- Confusion Matrix: {"true_positive": 19, "true_negative": 18, "false_positive": 7, "false_negative": 6}

### Chinese

- Accuracy: 0.4666666666666667
- Precision: 0.48148148148148145
- Recall: 0.8666666666666667
- Specificity: 0.06666666666666667
- F1: 0.6190476190476191
- Confusion Matrix: {"true_positive": 13, "true_negative": 1, "false_positive": 14, "false_negative": 2}

## Replacement Cases

| Dropped | Replacement | Truth | Pred | URL |
| --- | --- | --- | --- | --- |
| english__pitt__dementia__fluency__244_0 | english__pitt__dementia__sentence__283_0 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/283-0.mp3?f=save |
| english__pitt__dementia__fluency__094_3 | english__pitt__dementia__cookie__276_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/276-0.mp3?f=save |
| mandarin__chou__hc__003__003_park | mandarin__chou__hc__006__006_park | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/006/006_park.mp3?f=save |
| mandarin__chou__mci__103__103_park | mandarin__chou__mci__036__036_park | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/036/036_park.mp3?f=save |

## Case Table

| # | Group | Case ID | Truth | Pred | URL |
| --- | --- | --- | --- | --- | --- |
| 1 | english | english__pitt__control__cookie__280_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/280-0.mp3?f=save |
| 2 | english | english__pitt__control__cookie__015_4 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/015-4.mp3?f=save |
| 3 | english | english__lu__control__f50 | HC | HC | https://media.talkbank.org:443/dementia/English/Lu/Control//F50.mp3?f=save |
| 4 | english | english__pitt__control__cookie__054_0 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/054-0.mp3?f=save |
| 5 | english | english__pitt__control__cookie__143_3 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/143-3.mp3?f=save |
| 6 | english | english__pitt__control__cookie__114_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/114-0.mp3?f=save |
| 7 | english | english__pitt__control__cookie__059_4 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/059-4.mp3?f=save |
| 8 | english | english__pitt__control__cookie__295_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/295-0.mp3?f=save |
| 9 | english | english__pitt__control__cookie__210_2 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/210-2.mp3?f=save |
| 10 | english | english__pitt__control__cookie__137_2 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/137-2.mp3?f=save |
| 11 | english | english__pitt__control__cookie__211_2 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/211-2.mp3?f=save |
| 12 | english | english__pitt__control__cookie__304_1 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/304-1.mp3?f=save |
| 13 | english | english__pitt__control__cookie__182_3 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/182-3.mp3?f=save |
| 14 | english | english__pitt__control__cookie__245_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/245-0.mp3?f=save |
| 15 | english | english__lu__control__f46 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Lu/Control//F46.mp3?f=save |
| 16 | english | english__pitt__control__cookie__242_1 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/242-1.mp3?f=save |
| 17 | english | english__lu__control__f39 | HC | HC | https://media.talkbank.org:443/dementia/English/Lu/Control//F39.mp3?f=save |
| 18 | english | english__pitt__control__cookie__132_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/132-0.mp3?f=save |
| 19 | english | english__pitt__control__cookie__196_1 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/196-1.mp3?f=save |
| 20 | english | english__pitt__control__cookie__092_2 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/092-2.mp3?f=save |
| 21 | english | english__pitt__control__cookie__209_1 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/209-1.mp3?f=save |
| 22 | english | english__pitt__control__cookie__208_0 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/208-0.mp3?f=save |
| 23 | english | english__pitt__control__cookie__245_1 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/245-1.mp3?f=save |
| 24 | english | english__pitt__control__cookie__255_1 | HC | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/255-1.mp3?f=save |
| 25 | english | english__pitt__control__cookie__086_4 | HC | HC | https://media.talkbank.org:443/dementia/English/Pitt/Control/cookie/086-4.mp3?f=save |
| 26 | english | english__pitt__dementia__sentence__358_0 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/358-0.mp3?f=save |
| 27 | english | english__pitt__dementia__sentence__018_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/018-0.mp3?f=save |
| 28 | english | english__pitt__dementia__cookie__361_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/361-0.mp3?f=save |
| 29 | english | english__pitt__dementia__fluency__212_1 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/fluency/212-1.mp3?f=save |
| 30 | english | english__pitt__dementia__cookie__016_4 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/016-4.mp3?f=save |
| 31 | english | english__pitt__dementia__cookie__497_0 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/497-0.mp3?f=save |
| 32 | english | english__pitt__dementia__cookie__057_1 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/057-1.mp3?f=save |
| 33 | english | english__pitt__dementia__fluency__033_3 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/fluency/033-3.mp3?f=save |
| 34 | english | english__pitt__dementia__sentence__070_2 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/070-2.mp3?f=save |
| 35 | english | english__pitt__dementia__sentence__361_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/361-0.mp3?f=save |
| 36 | english | english__pitt__dementia__cookie__609_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/609-0.mp3?f=save |
| 37 | english | english__pitt__dementia__cookie__094_3 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/094-3.mp3?f=save |
| 38 | english | english__pitt__dementia__sentence__636_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/636-0.mp3?f=save |
| 39 | english | english__pitt__dementia__sentence__007_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/007-0.mp3?f=save |
| 40 | english | english__pitt__dementia__fluency__058_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/fluency/058-0.mp3?f=save |
| 41 | english | english__pitt__dementia__fluency__061_1 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/fluency/061-1.mp3?f=save |
| 42 | english | english__pitt__dementia__cookie__213_1 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/213-1.mp3?f=save |
| 43 | english | english__pitt__dementia__cookie__260_2 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/260-2.mp3?f=save |
| 44 | english | english__pitt__dementia__cookie__221_1 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/221-1.mp3?f=save |
| 45 | english | english__pitt__dementia__cookie__091_2 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/091-2.mp3?f=save |
| 46 | english | english__pitt__dementia__cookie__270_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/270-0.mp3?f=save |
| 47 | english | english__pitt__dementia__cookie__023_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/023-0.mp3?f=save |
| 48 | english | english__pitt__dementia__cookie__610_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/610-0.mp3?f=save |
| 49 | chinese | mandarin__chou__hc__022__022_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/022/022_market.mp3?f=save |
| 50 | chinese | mandarin__chou__hc__027__027_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/027/027_market.mp3?f=save |
| 51 | chinese | mandarin__chou__hc__002__002_park | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/002/002_park.mp3?f=save |
| 52 | chinese | mandarin__chou__hc__026__026_daddy | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/026/026_Daddy.mp3?f=save |
| 53 | chinese | mandarin__chou__hc__022__022_daddy | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/022/022_Daddy.mp3?f=save |
| 54 | chinese | mandarin__chou__hc__005__005_daddy | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/005/005_Daddy.mp3?f=save |
| 55 | chinese | mandarin__chou__hc__064__064_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/064/064_market.mp3?f=save |
| 56 | chinese | mandarin__chou__hc__031__031_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/031/031_market.mp3?f=save |
| 57 | chinese | mandarin__chou__hc__005__005_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/005/005_market.mp3?f=save |
| 58 | chinese | mandarin__chou__hc__018__018_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/018/018_market.mp3?f=save |
| 59 | chinese | mandarin__chou__hc__064__064_park | HC | HC | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/064/064_park.mp3?f=save |
| 60 | chinese | mandarin__chou__hc__002__002_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/002/002_market.mp3?f=save |
| 61 | chinese | mandarin__chou__hc__006__006_daddy | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/006/006_Daddy.mp3?f=save |
| 62 | chinese | mandarin__chou__hc__013__013_market | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/013/013_market.mp3?f=save |
| 63 | chinese | mandarin__chou__mci__046__046_market | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/046/046_market.mp3?f=save |
| 64 | chinese | mandarin__chou__mci__132__132_market | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/132/132_market.mp3?f=save |
| 65 | chinese | mandarin__chou__mci__006__006_daddy | cognitive_risk | HC | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/006/006_Daddy.mp3?f=save |
| 66 | chinese | mandarin__chou__mci__006__006_market | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/006/006_market.mp3?f=save |
| 67 | chinese | mandarin__chou__mci__150__150_market | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/150/150_market.mp3?f=save |
| 68 | chinese | mandarin__chou__mci__117__117_park | cognitive_risk | HC | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/117/117_park.mp3?f=save |
| 69 | chinese | mandarin__chou__mci__136__136_daddy | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/136/136_Daddy.mp3?f=save |
| 70 | chinese | mandarin__chou__mci__011__011_daddy | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/011/011_Daddy.mp3?f=save |
| 71 | chinese | mandarin__chou__mci__094__094_park | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/094/094_park.mp3?f=save |
| 72 | chinese | mandarin__chou__mci__164__164_park | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/164/164_park.mp3?f=save |
| 73 | chinese | mandarin__chou__mci__071__071_daddy | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/071/071_Daddy.mp3?f=save |
| 74 | chinese | mandarin__chou__mci__027__027_market | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/027/027_market.mp3?f=save |
| 75 | chinese | mandarin__chou__mci__036__036_daddy | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/036/036_Daddy.mp3?f=save |
| 76 | chinese | mandarin__chou__mci__071__071_park | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/071/071_park.mp3?f=save |
| 77 | english | english__pitt__dementia__sentence__283_0 | cognitive_risk | HC | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/sentence/283-0.mp3?f=save |
| 78 | english | english__pitt__dementia__cookie__276_0 | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/English/Pitt/Dementia/cookie/276-0.mp3?f=save |
| 79 | chinese | mandarin__chou__hc__006__006_park | HC | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/HC/006/006_park.mp3?f=save |
| 80 | chinese | mandarin__chou__mci__036__036_park | cognitive_risk | cognitive_risk | https://media.talkbank.org:443/dementia/Mandarin/Chou/MCI/036/036_park.mp3?f=save |
