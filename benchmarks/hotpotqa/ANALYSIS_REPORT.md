# HotpotQA Results Analysis Report

**Date:** 2026-03-17  
**Files analyzed:**
1. `results_fullwiki_validation_DEEP_qwen3.5-plus_20q_20260317_125859.json` (latest, **enable_thinking=false**)
2. `results_fullwiki_validation_DEEP_qwen3.5-plus_20q_20260317_124732.json` (second latest, **enable_thinking=true**)

---

## 1. Overview

| Metric | enable_thinking=false | enable_thinking=true |
|--------|----------------------|----------------------|
| **Total samples** | 20 | 20 |
| **Errors** | 0 | 0 |
| **Ans EM (from report)** | 40.0% | 55.0% |
| **Ans F1 (from report)** | 40.53% | 57.5% |
| **sp_em=0 %** | 90.0% (18/20) | 95.0% (19/20) |
| **Worst performers (ans_em=0, ans_f1<0.3)** | 12 | 8 |

---

## 2. Per-Sample Details

### File 1: enable_thinking=false (latest)

| _id | question (60 chars) | prediction (60 chars) | gold | ans_em | ans_f1 | sp_em | sp_f1 | elapsed | loops | total_tokens | error |
|-----|---------------------|-----------------------|------|--------|--------|-------|-------|--------|-------|--------------|-------|
| 5a7a567255429941d65f25bd | What was Iqbal F. Qadir on when he participated in an attack… | PNS Mangro | flotilla | 0 | 0.0 | 0.0 | 0.5 | 74.11 | 9 | 104445 | |
| 5abca1a55542993a06baf937 | When did the park at which Tivolis Koncertsal is located ope… | 15 August 1843 | 15 August 1843 | 1 | 1.0 | 0.0 | 0.8 | 18.77 | 3 | 13562 | |
| 5a73977d554299623ed4ac08 | What is the shared country of ancestry between Art Laboe and… | Armenia | Armenian | 0 | 0.0 | 0.0 | 0.8 | 24.45 | 4 | 8626 | |
| 5ab514c05542991779162d72 | The school in which the Wilmslow Show is designated … | comprehensive secondary school | Centre of Excellence | 0 | 0.0 | 1.0 | 1.0 | 47.44 | 8 | 50864 | |
| 5add2b435542990d50227e11 | Who will Billy Howle be seen opposite in the upcoming Britis… | Saoirse Ronan | Saoirse Ronan | 1 | 1.0 | 0.0 | 0.6667 | 38.89 | 6 | 66503 | |
| 5a88d6df554299206df2b377 | What animated movie, starring Danny Devito, featured music w… | The Lorax | The Lorax | 1 | 1.0 | 0.0 | 0.2857 | 68.91 | 6 | 46553 | |
| 5ae6b6065542991bbc976168 | Out of the actors who have played the role of Luc Deveraux i… | Alastair Mackenzie | Scott Adkins | 0 | 0.0 | 0.0 | 0.0 | 97.54 | 10 | 116219 | |
| 5ae531ee5542990ba0bbb1ff | Tommy's Honour was a drama film that included the actor who … | War & Peace | War & Peace | 1 | 1.0 | 0.0 | 0.0 | 37.97 | 7 | 53727 | |
| 5a8aa5835542996c9b8d5f4e | Which rock band chose its name by drawing it out of a hat, S… | Switchfoot | Midnight Oil | 0 | 0.0 | 0.0 | 0.0 | 26.47 | 4 | 14301 | |
| 5ab82d095542990e739ec853 | "Tunak", is a bhangra/pop love song by an artist born in whi… | 1967 | 1967 | 1 | 1.0 | 1.0 | 1.0 | 17.6 | 3 | 5828 | |
| 5ae4c01e55429913cc2044f3 | Which Captain launched the attack which led to more casualti… | Captain John Underhill | Captain John Underhill | 1 | 1.0 | 0.0 | 0.75 | 38.3 | 7 | 67711 | |
| 5a89fea655429970aeb701eb | In which film did Emilio Estevez star in in the same year as… | The Outsiders | The Outsiders | 1 | 1.0 | 0.0 | 0.5 | 27.28 | 4 | 16947 | |
| 5a80cf4c55429938b61421f6 | What was the concept of the business Eric S .Pistorius worke… | a law firm specializing in personal injury, litigation, crim… | to ensure wide visibility and understanding of cases in a region | 0 | 0.1053 | 0.0 | 0.3636 | 142.14 | 6 | 145699 | |
| 5a89b1de5542992e4fca8378 | Which port city lies approximately 25 km north of the Lingna… | unknown | Keelung | 0 | 0.0 | 0.0 | 0.2 | 59.97 | 5 | 50079 | |
| 5a8778d25542994846c1cd89 | Has Stefan Edberg won more events than Édouard Roger-Vassel… | Yes | yes | 1 | 1.0 | 0.0 | 0.6154 | 88.94 | 9 | 301275 | |
| 5a77897f55429949eeb29edc | Jason Regler, stated that he had the idea for the flashing … | guitar | an organ | 0 | 0.0 | 0.0 | 0.3333 | 25.85 | 5 | 149979 | |
| 5ae0132d55429925eb1afc00 | The Soul of Buddha is a 1918 American silent romance film s… | Staten Island Ferry | the George Washington Bridge | 0 | 0.0 | 0.0 | 0.6667 | 54.51 | 5 | 45018 | |
| 5a7129685542994082a3e5fa | Which "Blackzilians" fighter is currently competing in the M… | Yoel Romero | Vitor Belfort | 0 | 0.0 | 0.0 | 0.0 | 36.23 | 6 | 42901 | |
| 5ae762835542997b22f6a711 | Were was the Mexican state after which there is Villa Unión,… | Mexico | tip of the Baja California | 0 | 0.0 | 0.0 | 0.0 | 29.16 | 4 | 8864 | |
| 5ae2f5b955429928c423957e | What language, traditionally written with the ancient Libyco… | Berber | The Tugurt language | 0 | 0.0 | 0.0 | 0.5 | 35.67 | 7 | 64714 | |

### File 2: enable_thinking=true (second latest)

| _id | question (60 chars) | prediction (60 chars) | gold | ans_em | ans_f1 | sp_em | sp_f1 | elapsed | loops | total_tokens | error |
|-----|---------------------|-----------------------|------|--------|--------|-------|-------|--------|-------|--------------|-------|
| 5a7a567255429941d65f25bd | What was Iqbal F. Qadir on when he participated in an attack… | unknown | flotilla | 0 | 0.0 | 0.0 | 0.4 | 106.4 | 5 | 102471 | |
| 5abca1a55542993a06baf937 | When did the park at which Tivolis Koncertsal is located ope… | 15 August 1843 | 15 August 1843 | 1 | 1.0 | 0.0 | 0.6667 | 19.7 | 3 | 13616 | |
| 5a73977d554299623ed4ac08 | What is the shared country of ancestry between Art Laboe and… | Armenia | Armenian | 0 | 0.0 | 0.0 | 0.8 | 30.45 | 4 | 8821 | |
| 5ab514c05542991779162d72 | The school in which the Wilmslow Show is designated … | comprehensive secondary school | Centre of Excellence | 0 | 0.0 | 0.0 | 0.6667 | 39.24 | 7 | 60168 | |
| 5add2b435542990d50227e11 | Who will Billy Howle be seen opposite in the upcoming Britis… | Saoirse Ronan | Saoirse Ronan | 1 | 1.0 | 0.0 | 0.8 | 46.66 | 8 | 132378 | |
| 5a88d6df554299206df2b377 | What animated movie, starring Danny Devito, featured music w… | The Lorax | The Lorax | 1 | 1.0 | 0.0 | 0.8 | 54.52 | 6 | 104707 | |
| 5ae6b6065542991bbc976168 | Out of the actors who have played the role of Luc Deveraux i… | Scott Adkins | Scott Adkins | 1 | 1.0 | 0.0 | 0.5714 | 63.6 | 10 | 122101 | |
| 5ae531ee5542990ba0bbb1ff | Tommy's Honour was a drama film that included the actor who … | War & Peace | War & Peace | 1 | 1.0 | 0.0 | 0.5 | 66.54 | 10 | 89757 | |
| 5a8aa5835542996c9b8d5f4e | Which rock band chose its name by drawing it out of a hat, S… | Midnight Oil | Midnight Oil | 1 | 1.0 | 0.0 | 0.3333 | 36.28 | 4 | 14064 | |
| 5ab82d095542990e739ec853 | "Tunak", is a bhangra/pop love song by an artist born in whi… | 1967 | 1967 | 1 | 1.0 | 1.0 | 1.0 | 39.8 | 6 | 40798 | |
| 5ae4c01e55429913cc2044f3 | Which Captain launched the attack which led to more casualti… | Captain John Underhill | Captain John Underhill | 1 | 1.0 | 0.0 | 0.8571 | 41.35 | 7 | 67155 | |
| 5a89fea655429970aeb701eb | In which film did Emilio Estevez star in in the same year as… | unknown | The Outsiders | 0 | 0.0 | 0.0 | 0.0 | 161.0 | 4 | 119608 | |
| 5a80cf4c55429938b61421f6 | What was the concept of the business Eric S .Pistorius worke… | unknown | to ensure wide visibility and understanding of cases in a region | 0 | 0.0 | 0.0 | 0.8 | 156.25 | 6 | 158545 | |
| 5a89b1de5542992e4fca8378 | Which port city lies approximately 25 km north of the Lingna… | Keelung | Keelung | 1 | 1.0 | 0.0 | 0.4 | 37.55 | 5 | 41675 | |
| 5a8778d25542994846c1cd89 | Has Stefan Edberg won more events than Édouard Roger-Vassel… | Yes | yes | 1 | 1.0 | 0.0 | 0.4615 | 106.25 | 6 | 181209 | |
| 5a77897f55429949eeb29edc | Jason Regler, stated that he had the idea for the flashing … | xylophone | an organ | 0 | 0.0 | 0.0 | 0.6667 | 32.25 | 5 | 36770 | |
| 5ae0132d55429925eb1afc00 | The Soul of Buddha is a 1918 American silent romance film s… | the George Washington Bridge | the George Washington Bridge | 1 | 1.0 | 0.0 | 0.6667 | 69.94 | 6 | 66929 | |
| 5a7129685542994082a3e5fa | Which "Blackzilians" fighter is currently competing in the M… | Yoel Romero | Vitor Belfort | 0 | 0.0 | 0.0 | 0.0 | 72.43 | 10 | 138352 | |
| 5ae762835542997b22f6a711 | Were was the Mexican state after which there is Villa Unión,… | northwestern Mexico, along the Pacific coast | tip of the Baja California | 0 | 0.0 | 0.0 | 0.2 | 84.38 | 3 | 127600 | |
| 5ae2f5b955429928c423957e | What language, traditionally written with the ancient Libyco… | The Berber language | The Tugurt language | 0 | 0.5 | 0.0 | 0.0 | 120.56 | 7 | 240808 | |

---

## 3. Worst Performing Samples (ans_em=0, ans_f1<0.3)

### enable_thinking=false — 12 samples

| _id | question (60 chars) | prediction | gold | ans_f1 |
|-----|---------------------|------------|------|-------|
| 5a7a567255429941d65f25bd | What was Iqbal F. Qadir on when he participated in an attack… | PNS Mangro | flotilla | 0.0 |
| 5a73977d554299623ed4ac08 | What is the shared country of ancestry between Art Laboe and… | Armenia | Armenian | 0.0 |
| 5ab514c05542991779162d72 | The school in which the Wilmslow Show is designated … | comprehensive secondary school | Centre of Excellence | 0.0 |
| 5ae6b6065542991bbc976168 | Out of the actors who have played the role of Luc Deveraux i… | Alastair Mackenzie | Scott Adkins | 0.0 |
| 5a8aa5835542996c9b8d5f4e | Which rock band chose its name by drawing it out of a hat, S… | Switchfoot | Midnight Oil | 0.0 |
| 5a80cf4c55429938b61421f6 | What was the concept of the business Eric S .Pistorius worke… | a law firm specializing in personal injury… | to ensure wide visibility and understanding of cases in a region | 0.1053 |
| 5a89b1de5542992e4fca8378 | Which port city lies approximately 25 km north of the Lingna… | unknown | Keelung | 0.0 |
| 5a77897f55429949eeb29edc | Jason Regler, stated that he had the idea for the flashing … | guitar | an organ | 0.0 |
| 5ae0132d55429925eb1afc00 | The Soul of Buddha is a 1918 American silent romance film s… | Staten Island Ferry | the George Washington Bridge | 0.0 |
| 5a7129685542994082a3e5fa | Which "Blackzilians" fighter is currently competing in the M… | Yoel Romero | Vitor Belfort | 0.0 |
| 5ae762835542997b22f6a711 | Were was the Mexican state after which there is Villa Unión,… | Mexico | tip of the Baja California | 0.0 |
| 5ae2f5b955429928c423957e | What language, traditionally written with the ancient Libyco… | Berber | The Tugurt language | 0.0 |

### enable_thinking=true — 8 samples

| _id | question (60 chars) | prediction | gold | ans_f1 |
|-----|---------------------|------------|------|-------|
| 5a7a567255429941d65f25bd | What was Iqbal F. Qadir on when he participated in an attack… | unknown | flotilla | 0.0 |
| 5a73977d554299623ed4ac08 | What is the shared country of ancestry between Art Laboe and… | Armenia | Armenian | 0.0 |
| 5ab514c05542991779162d72 | The school in which the Wilmslow Show is designated … | comprehensive secondary school | Centre of Excellence | 0.0 |
| 5a89fea655429970aeb701eb | In which film did Emilio Estevez star in in the same year as… | unknown | The Outsiders | 0.0 |
| 5a80cf4c55429938b61421f6 | What was the concept of the business Eric S .Pistorius worke… | unknown | to ensure wide visibility and understanding of cases in a region | 0.0 |
| 5a77897f55429949eeb29edc | Jason Regler, stated that he had the idea for the flashing … | xylophone | an organ | 0.0 |
| 5a7129685542994082a3e5fa | Which "Blackzilians" fighter is currently competing in the M… | Yoel Romero | Vitor Belfort | 0.0 |
| 5ae762835542997b22f6a711 | Were was the Mexican state after which there is Villa Unión,… | northwestern Mexico, along the Pacific coast | tip of the Baja California | 0.0 |

---

## 4. Failure Pattern Analysis

### Common patterns among worst performers

1. **Near-synonym / related-entity confusion**
   - *Armenia* vs *Armenian* (country vs demonym)
   - *Berber* vs *The Tugurt language* (broader category vs specific)
   - *comprehensive secondary school* vs *Centre of Excellence* (descriptor vs official designation)

2. **Wrong entity from same domain**
   - *Alastair Mackenzie* vs *Scott Adkins* (both actors who played Luc Deveraux)
   - *Yoel Romero* vs *Vitor Belfort* (both Blackzilians fighters)
   - *Switchfoot* vs *Midnight Oil* (both rock bands)
   - *Staten Island Ferry* vs *the George Washington Bridge* (both NYC landmarks)

3. **Over-specific or wrong extraction**
   - *PNS Mangro* vs *flotilla* (specific ship vs vessel type)
   - *guitar* vs *an organ* (wrong instrument)
   - *Mexico* vs *tip of the Baja California* (country vs specific location)

4. **Long/phrase answers**
   - *a law firm specializing in personal injury…* vs *to ensure wide visibility and understanding of cases in a region*
   - Model tends to extract a concrete entity instead of a conceptual description.

5. **Unknown / retrieval failure**
   - *unknown* when gold is a specific entity (Keelung, The Outsiders, flotilla)
   - Suggests retrieval did not surface the right evidence.

### enable_thinking=true improvements

- **5ae6b606**: Alastair Mackenzie → **Scott Adkins** ✓
- **5a8aa583**: Switchfoot → **Midnight Oil** ✓
- **5a89b1de**: unknown → **Keelung** ✓
- **5ae0132d**: Staten Island Ferry → **the George Washington Bridge** ✓
- **5ae2f5b955429928c423957e**: Berber → The Berber language (ans_f1 0.5, still wrong gold “The Tugurt language”)

### Persistent failures (both runs)

- *Armenia* vs *Armenian* (normalization / granularity)
- *comprehensive secondary school* vs *Centre of Excellence*
- *Yoel Romero* vs *Vitor Belfort*
- *guitar* vs *an organ*
- *Mexico* vs *tip of the Baja California*
- Iqbal F. Qadir / flotilla (retrieval or evidence quality)

---

## 5. sp_em=0 Statistics

| Config | % with sp_em=0 | Count | Avg sp_f1 (when sp_em=0) | sp_f1 values |
|--------|----------------|-------|---------------------------|--------------|
| **enable_thinking=false** | 90.0% | 18/20 | 0.3879 | 0.5, 0.8, 0.8, 0.6667, 0.2857, 0.0, 0.0, 0.0, 0.75, 0.5, 0.3636, 0.2, 0.6154, 0.3333, 0.6667, 0.0, 0.0, 0.5 |
| **enable_thinking=true** | 95.0% | 19/20 | 0.5047 | 0.4, 0.6667, 0.8, 0.6667, 0.8, 0.8, 0.5714, 0.5, 0.3333, 0.8571, 0.0, 0.8, 0.4, 0.4615, 0.6667, 0.6667, 0.0, 0.2, 0.0 |

**Interpretation:** Most samples have sp_em=0 (supporting facts not exactly matched). With thinking enabled, sp_f1 for those samples is higher (0.50 vs 0.39), so partial supporting-fact overlap improves even when full match is rare.

---

## 6. Elapsed Time Distribution

| Config | min (s) | max (s) | median (s) | Outliers |
|--------|---------|---------|------------|----------|
| **enable_thinking=false** | 17.6 | 142.14 | 38.3 | 142.14 (>1.5×IQR above Q3) |
| **enable_thinking=true** | 19.7 | 161.0 | 63.6 | — |

**Summary:**
- enable_thinking=true is slower: median 63.6s vs 38.3s (~66% increase).
- Max latency: 161.0s (thinking) vs 142.14s (no thinking).
- Single outlier in no-thinking run: 142.14s (5a80cf4c – Eric S. Pistorius concept question).

---

## 7. Summary Table

| Metric | enable_thinking=false | enable_thinking=true |
|--------|----------------------|----------------------|
| Total samples | 20 | 20 |
| Errors | 0 | 0 |
| Ans EM | 40.0% | 55.0% |
| Ans F1 | 40.53% | 57.5% |
| sp_em=0 % | 90.0% | 95.0% |
| Avg sp_f1 (sp_em=0) | 0.39 | 0.50 |
| Worst performers | 12 | 8 |
| Elapsed min (s) | 17.6 | 19.7 |
| Elapsed max (s) | 142.14 | 161.0 |
| Elapsed median (s) | 38.3 | 63.6 |
| Avg tokens | 68,691 | 93,377 |
