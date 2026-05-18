import sys

with open("thesis.txt", "r") as f: 
    text = f.read()

PERCENT = 0.2



if __name__ == "__main__":
    if len(sys.argv) < 2:
        percent = PERCENT
    elif len(sys.argv) == 2:
        try:
            percent = float(sys.argv[1])
        except Exception as _: 
            percent = PERCENT
    else: 
        percent = PERCENT 
    print("CHARS: ", len(text))
    print("NORMOSTRANY: ", ns:=(len(text) / 1800))
    var = percent*ns
    print(f"NORMOSTRANY +- {int(percent*100)} %: {ns}+-{var}")
    print(f"    {ns-var}-{ns+var}")

