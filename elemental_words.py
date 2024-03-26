def get_elements_dict():
    from periodictable import elements

    old = {
        113: ("Uut", "Ununtrium"),
        115: ("Uup", "Ununpentium"),
        117: ("Uus", "Ununseptium"),
        118: ("Uuo", "Ununoctium"),
    }
    elems = {e.symbol: e.name for e in elements}
    for n in old:
        del elems[elements[n].symbol]
        symbol, name = old[n]
        elems[symbol] = name
    return {s.lower(): n for s, n in elems.items()}


def elemental_forms(word):
    ELEMENTS = get_elements_dict()

    def _memoized(w, memo):
        if w not in memo:
            prefixes = [s for s in ELEMENTS if w.startswith(s)]
            forms = []
            for p in prefixes:
                if w == p:
                    forms.append([p])
                else:
                    tail_forms = _memoized(w.removeprefix(p), memo)
                    forms.extend([[p, *form] for form in tail_forms])
            memo[w] = forms
        return memo[w]

    found = {}
    forms = _memoized(word.lower(), found)

    return [
        [f"{ELEMENTS[s].capitalize()} ({s.capitalize()})" for s in form]
        for form in forms
    ]
