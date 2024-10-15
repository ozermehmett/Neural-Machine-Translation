import re
from utils import unicodeToAscii


def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([!.?])", r" \1", s)
  s = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ?.!]+", " ", s)
  s = re.sub(r"(cc by [^.]+|attribution[^.]+|tatoeba\.org[^.]+)", "", s)
  return s
