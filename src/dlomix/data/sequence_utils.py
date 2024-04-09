# can be completely removed after moving everything to processors
import re


def rebuild_proforma_sequence(seq, mod, n_term, c_term, no_mod_value=-1):
    proforma_sequence = [
        f"{s}[UNIMOD:{m}]" if m != no_mod_value else s for s, m in zip(seq, mod)
    ]

    if n_term != no_mod_value:
        proforma_sequence.insert(0, f"[UNIMOD:{n_term}]-")

    if c_term != no_mod_value:
        proforma_sequence.append(f"-[UNIMOD:{c_term}]")
    return proforma_sequence


def parse_sequence_native(sequence_string):
    splitted = sequence_string.split("-")

    if len(splitted) == 1:
        n_term, seq, c_term = "[]-", splitted[0], "-[]"
    elif len(splitted) == 2:
        if splitted[0].startswith("[UNIMOD:"):
            n_term, seq, c_term = splitted[0] + "-", splitted[1], "-[]"
        else:
            n_term, seq, c_term = "[]-", splitted[0], "-" + splitted[1]
    elif len(splitted) == 3:
        n_term, seq, c_term = splitted
        n_term += "-"
        c_term = "-" + c_term

    seq = re.findall(r"[A-Za-z](?:\[UNIMOD:\d+\])?|[^\[\]]", seq)
    return n_term, seq, c_term
