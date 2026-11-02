"""Settings of forcearray that you may want to edit.

* ``LS_BANDLIST`` / ``S2_BANDLIST``: band name -> band number (1-based) in
  the Landsat / Sentinel-2 BOA files. The names are what you pass in
  ``bandlist``.
* ``QAI_FILTER_DEFAULT`` / ``QAI_FILTER_BEST``: which QAI values count as
  valid (``best_cso=False`` / ``best_cso=True``). Each entry is one QAI flag,
  in bit order from bit 0 upwards, with the allowed bit values written as
  strings ('0', '1' for 1-bit flags, '00'...'11' for 2-bit flags). For a
  2-bit flag the first character is the LOWER bit, e.g. for 'Cloud state'
  (bits 1-2) '01' means bit 1 = 0 and bit 2 = 1. A pixel is valid if every
  flag has an allowed value. Keep all 13 entries, in this order, so the bits
  add up to 16.
"""

LS_BANDLIST = {
    'BLU': 1,
    'GRN': 2,
    'RED': 3,
    'NIR': 4,
    'SW1': 5,
    'SW2': 6,
}

S2_BANDLIST = {
    'BLU': 1,
    'GRN': 2,
    'RED': 3,
    'RE1': 4,
    'RE2': 5,
    'RE3': 6,
    'BNR': 7,
    'NIR': 8,
    'SW1': 9,
    'SW2': 10,
}

QAI_FILTER_DEFAULT = {
    'Valid data': ['0'],
    'Cloud state': ['00'],
    'Cloud shadow flag': ['0'],
    'Snow flag': ['0'],
    'Water flag': ['0', '1'],
    'Aerosol state': ['00', '01', '10', '11'],
    'Subzero flag': ['0'],
    'Saturation flag': ['0'],
    'High sun zenith flag': ['0', '1'],
    'Illumination state': ['00', '01', '10', '11'],
    'Slope flag': ['0', '1'],
    'Water vapor flag': ['0', '1'],
    'Empty': ['0'],
}

QAI_FILTER_BEST = {
    'Valid data': ['0'],
    'Cloud state': ['00'],
    'Cloud shadow flag': ['0'],
    'Snow flag': ['0'],
    'Water flag': ['0', '1'],
    'Aerosol state': ['00'],
    'Subzero flag': ['0'],
    'Saturation flag': ['0'],
    'High sun zenith flag': ['0'],
    'Illumination state': ['00'],
    'Slope flag': ['0', '1'],
    'Water vapor flag': ['0', '1'],
    'Empty': ['0'],
}
