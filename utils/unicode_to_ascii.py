import unicodedata


def unicodeToAscii(s):
  return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" or c in "çğıöşüÇĞİÖŞÜ")
