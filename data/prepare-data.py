import json

if __name__ == "__main__":
    eng = open("small.eng").readlines()
    eng = [e.rstrip() for e in eng]
    ces = open("small.ces").readlines()
    ces = [c.rstrip() for c in ces]

    for src, tgt in zip(ces, eng):
        print(
            json.dumps(
                {
                    "src_lang": "Czech",
                    "tgt_lang": "English",
                    "src": src,
                    "tgt": tgt,
                },
                ensure_ascii=False,
            )
        )
