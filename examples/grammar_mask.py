"""Finite-state teaching model for grammar masks; not XGrammar or a GPU kernel."""
from dataclasses import dataclass

TRANSITIONS = {
    0: {"{": 1},
    1: {'"ok"': 2},
    2: {":": 3},
    3: {"true": 4, "false": 4},
    4: {",": 5},
    5: {'"count"': 6},
    6: {":": 7},
    7: {"0": 8, "1": 8, "2": 8},
    8: {"}": 9},
    9: {},
}
VOCAB = tuple(sorted({token for state in TRANSITIONS.values() for token in state}))


@dataclass
class ToyGrammar:
    state: int = 0

    def __post_init__(self):
        self.history = [self.state]

    def allowed_tokens(self):
        return frozenset(TRANSITIONS[self.state])

    def token_mask(self):
        allowed = self.allowed_tokens()
        return {token: token in allowed for token in VOCAB}

    def accept(self, token):
        target = TRANSITIONS[self.state].get(token)
        if target is None:
            return False
        self.state = target
        self.history.append(target)
        return True

    def rollback(self, count=1):
        if count < 0 or count >= len(self.history):
            raise ValueError("rollback count must preserve the initial state")
        if count:
            self.history = self.history[:-count]
            self.state = self.history[-1]

    @property
    def terminated(self):
        return self.state == 9


class GrammarCache:
    def __init__(self):
        self._items = {}
        self.compiles = 0

    def get(self, key):
        if key not in self._items:
            self._items[key] = tuple(sorted(VOCAB))
            self.compiles += 1
        return self._items[key]


if __name__ == "__main__":
    grammar = ToyGrammar()
    output = ['{', '"ok"', ':', 'true', ',', '"count"', ':', '2', '}']
    for token in output:
        assert token in grammar.allowed_tokens()
        assert grammar.accept(token)
    assert grammar.terminated
    print("valid toy JSON token sequence:", output)
