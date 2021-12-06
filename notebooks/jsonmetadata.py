# %%
import json
import os
from pathlib import Path
import multiprocessing

# %%
# fmt:off
isocodes = [ "en", "de", "fr", "ca", "rw", "es", "ru", "it", "pl", "eu",
    "fa", "nl", "eo", "pt", "cy", "tt", "cs", "uk", "et", "tr",
    "mn", "ky", "ar", "fy-NL", "sv-SE", "mt", "id", "el", "br",
    "rm-sursilv", "ro", "sl", "sah", "lv", "ia", "sk", "cv", "ga-IE",
    "zh-CN", "ka", "cnh", "ha", "rm-vallader", "ta", "vi", "as",
    "gn", "or", "dv", "lt", ]
print(len(isocodes))

l2i = { "German": "de", "English": "en", "French": "fr", "Catalan": "ca", "Kinyarwada": "rw", "Spanish": "es", "Russian": "ru", "Italian": "it", "Polish": "pl",
    "Basque": "eu", "Persian": "fa", "Dutch": "nl", "Esparanto": "eo", "Portuguese": "pt", "Welsh": "cy", "Tatar": "tt", "Czech": "cs", "Ukranian": "uk", "Estonian": "et",
    "Turkish": "tr", "Mongolian": "mn", "Kyrgyz": "ky", "Arabic": "ar", "Frisian": "fy-NL", "Swedish": "sv-SE", "Maltese": "mt", "Indonesian": "id", "Greek": "el", "Breton": "br",
    "Sursilvan": "rm-sursilv", "Romanian": "ro", "Slovenian": "sl", "Sakha": "sah", "Latvian": "lv", "Interlingua": "ia", "Slovak": "sk",
    "Chuvash": "cv", "Irish": "ga-IE", "Chinese": "zh-CN", "Georgian": "ka", "Hakha Chin": "cnh", "Hausa": "ha", "Vallader": "rm-vallader", "Tamil": "ta", "Vietnamese": "vi",
    "Assamese": "as", "Guarani": "gn", "Oriya": "or", "Dhivehi": "dv", "Lithuanian": "lt",
}
print(len(l2i))
# fmt:on

i2l = {v:k for k,v in l2i.items()}
print(i2l)

# %%
basedir = Path("/mnt/disks/std3/opus/generated/common_voice/frequent_words")

def metadata_lang(code_mdict):
    code, metadata = code_mdict
    clips = basedir / code / "clips"
    words = os.listdir(clips)
    n_words = len(words)
    language_data = {}
    language_data["language"] = i2l[code]
    language_data["number_of_words"] = n_words
    wordcounts = {}
    language_data["wordcounts"] = wordcounts
    filenames = {}
    language_data["filenames"] = filenames
    for word in words:
        samples = os.listdir(clips / word)
        wordcounts[word] = len(samples)
        filenames[word] = samples
    # this must happen last for the managed dict to contain all updates
    # from an unmanaged dict
    metadata[code] = language_data
    print(f"{code}:done")


with multiprocessing.Pool() as pool:
    with multiprocessing.Manager() as manager:
        metadata = manager.dict()
        to_process = [(code, metadata) for code in isocodes]
        for _ in pool.imap_unordered(metadata_lang, to_process):
            pass

        print('complete')
        json_metadata = metadata.copy()
        json_metadata["version"] = "version 1.0, Multilingual Spoken Words Corpus, https://mlcommons.org/en/multilingual-spoken-words"
        with open("metadata.json", 'w') as fh:
            json.dump(json_metadata, fh)

# %%
