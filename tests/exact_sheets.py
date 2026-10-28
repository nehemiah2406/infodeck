"""Three small made-up calculation sheets (fictional data) with different layouts, and questions about
them, for the exact-value evaluation (tests/exact_eval.py):
    A  the value on the same line as the item
    B  the value on the next line (a formula line), plus helper lines and a comparison line
    C  tables (local "Header: value; ..." rows and a Markdown table)
"""

SHEETS = {
    "CS-101_Pump_Skid_FICTIONAL.txt": """\
CALCULATION SHEET CS-101 (FICTIONAL) - Pump skid foundation
1. Loads
(1) Dead load, Wd = 42.5 kN
(2) Live load, Wl = 18 kN
(3) Wind load on skid, Ww = 6.2 kN
2. Anchor bolts
(1) Bolt diameter, d = 24 mm
(2) Tensile stress corresponding to Wd, N/mm²   σt = Wd / As = 96.4 N/mm²
(3) Embedment length, Le = 380 mm
(4) Number of anchor bolts, n = 8
3. Base plate
(1) Plate thickness, tp = 30 mm
(2) Bearing pressure under the plate, N/mm²   qb = (Wd + Wl) / Ab = 1.85 N/mm²
""",
    "CS-202_Davit_Arm_FICTIONAL.txt": """\
CALCULATION SHEET CS-202 (FICTIONAL) - Davit arm check
3. Arm geometry
(1) Arm length, La = 2,450 mm
(2) Bending moment corresponding to La, kN.m
Mb = SWL x La x 9.81 / 1000 = 36.06 kN.m
(3) Section modulus of the arm, (Z), cm³
Z = I / y = 512.7 cm³
4. Bending stress of arm, (σb), N/mm²
σb = Mb / Z = 70.3 N/mm²
where, k = (La / h) = 8.75
∴ σb = 70.3 N/mm² ≤ σa = 160 N/mm²
5. Allowable stress
(1) Allowable bending stress, σa = 160 N/mm²
(2) Safe working load, SWL = 1.5 t
""",
    "CS-303_Hatch_Cleats_FICTIONAL.txt": """\
CALCULATION SHEET CS-303 (FICTIONAL) - Hatch cover cleats
1. Cleat schedule
Cleat: C1; Preload (kN): 45; Bolt size: M20; Spacing (mm): 600
Cleat: C2; Preload (kN): 60; Bolt size: M24; Spacing (mm): 750
Cleat: C3; Preload (kN): 38; Bolt size: M16; Spacing (mm): 500
2. Rubber packing
| Packing | Compression (mm) | Hardness (Shore A) |
|---|---|---|
| P1 | 12 | 60 |
| P2 | 15 | 70 |
""",
}

# Same fields as data/eval/exact_questions.json; page 0 = text files have no page numbers.
QUESTIONS = [
    # A: same line
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "1.(1)", "question": "What is the dead load Wd?",
     "value": "42.5", "unit": "kN", "lines": ["(1) Dead load"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "1.(2)", "question": "What is the live load?",
     "value": "18", "unit": "kN", "lines": ["(2) Live load"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "1.(3)", "question": "What is the wind load on the skid?",
     "value": "6.2", "unit": "kN", "lines": ["(3) Wind load on skid"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "2.(1)", "question": "What is the anchor bolt diameter?",
     "value": "24", "unit": "mm", "lines": ["(1) Bolt diameter"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "2.(2)",
     "question": "What is the tensile stress corresponding to Wd?", "value": "96.4", "unit": "N/mm²",
     "lines": ["(2) Tensile stress corresponding to Wd"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "2.(3)", "question": "What is the embedment length?",
     "value": "380", "unit": "mm", "lines": ["(3) Embedment length"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "2.(4)", "question": "How many anchor bolts are there?",
     "value": "8", "lines": ["(4) Number of anchor bolts"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "3.(1)", "question": "What is the base plate thickness?",
     "value": "30", "unit": "mm", "lines": ["(1) Plate thickness"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "3.(2)",
     "question": "What is the bearing pressure under the base plate?", "value": "1.85", "unit": "N/mm²",
     "lines": ["(2) Bearing pressure under the plate"]},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "trap",
     "question": "What is the wind load corresponding to Wd?", "value": "", "note": "Wd is only a reference"},
    {"doc": "CS-101_Pump_Skid_FICTIONAL.txt", "page": 0, "item": "trap",
     "question": "What is the bolt preload for Wd?", "value": "", "note": "no such item: never 42.5"},
    # B: next line
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "3.(1)", "question": "What is the arm length La?",
     "value": "2,450", "unit": "mm", "lines": ["(1) Arm length"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "3.(2)",
     "question": "What is the bending moment corresponding to La?", "value": "36.06", "unit": "kN.m",
     "lines": ["Mb = SWL x La x 9.81 / 1000 = 36.06 kN.m"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "3.(3)",
     "question": "What is the section modulus of the arm?", "value": "512.7", "unit": "cm³",
     "lines": ["Z = I / y = 512.7 cm³"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "4.", "question": "What is the bending stress of the arm?",
     "value": "70.3", "unit": "N/mm²", "lines": ["σb = Mb / Z = 70.3 N/mm²"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "4. k", "question": "What is k = La / h?",
     "value": "8.75", "lines": ["k = (La / h) = 8.75"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "5.(1)", "question": "What is the allowable bending stress σa?",
     "value": "160", "unit": "N/mm²", "lines": ["(1) Allowable bending stress", "σa = 160 N/mm²"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "5.(2)", "question": "What is the safe working load SWL?",
     "value": "1.5", "unit": "t", "lines": ["(2) Safe working load"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "trap", "question": "What is σb?",
     "value": "70.3", "unit": "N/mm²", "lines": ["σb = Mb / Z = 70.3 N/mm²", "σb = 70.3 N/mm²"],
     "note": "never 160 from the comparison line"},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "trap",
     "question": "What is the section modulus corresponding to La?", "value": "512.7", "unit": "cm³",
     "lines": ["Z = I / y = 512.7 cm³"]},
    {"doc": "CS-202_Davit_Arm_FICTIONAL.txt", "page": 0, "item": "trap",
     "question": "What is the shear force corresponding to La?", "value": "", "note": "no shear force item: never 2,450"},
    # C: tables
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "row", "question": "What is the preload of cleat C2?",
     "value": "60", "lines": ["Cleat: C2"]},
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "row", "question": "What is the spacing of cleat C3?",
     "value": "500", "lines": ["Cleat: C3"]},
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "row", "question": "What is the preload of cleat C1?",
     "value": "45", "lines": ["Cleat: C1"]},
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "row", "question": "What is the compression of packing P2?",
     "value": "15", "lines": ["| P2 | 15 | 70 |"]},
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "row", "question": "What is the hardness of packing P1?",
     "value": "60", "lines": ["| P1 | 12 | 60 |"]},
    {"doc": "CS-303_Hatch_Cleats_FICTIONAL.txt", "page": 0, "item": "trap", "question": "What is the preload of cleat C4?",
     "value": "", "note": "no cleat C4"},
]
