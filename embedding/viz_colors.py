import seaborn as sns

iso2lang = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ky": "Kyrgyz",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "ta": "Tamil",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukranian",
}


def iso2line(isocode):
    if isocode in ["ca", "nl", "it", "en", "de", "es", "fr", "rw", "fa"]:
        return "dashed"
    return "solid"


# slightly more than 2 cycles of this color palette
color_list = [
    sns.color_palette("bright")[ix % len(sns.color_palette("bright"))]
    for ix in range(len(iso2lang.keys()))
]

iso2color_ix = {
    "ar": 22,
    "ca": 21,
    "cs": 20,
    "cy": 19,
    "de": 18,
    "en": 17,
    "es": 16,
    "et": 15,
    "eu": 14,
    "fa": 13,
    "fr": 12,
    "id": 11,
    "it": 10,
    "ky": 8,
    "nl": 9,
    "pl": 7,
    "pt": 6,
    "ru": 5,
    "rw": 4,
    "ta": 3,
    "tr": 2,
    "tt": 1,
    "uk": 0,
}


def iso2color(isocode):
    return color_list[iso2color_ix[isocode]]