"""BERT uncased WordPiece tokenizer, plus Grounding DINO's text mask construction.

Mirrors `BertTokenizer(do_lower_case=True)` for ASCII input: whitespace splitting,
lowercasing, punctuation splitting, then greedy longest-match-first WordPiece against
`vocab.txt`. Accent stripping and the CJK spacing rule are **not** implemented -- the
Grounding DINO prompt convention is lowercase ASCII phrases joined by " . " and ending
with ".", which this covers exactly.

`text_masks` reproduces `generate_masks_with_special_tokens_and_transfer_map`: each
phrase (the span between two special tokens) becomes its own attention block, and
position ids restart inside every phrase.
"""

from .tensor import Tensor

comptime CLS_ID = 101
comptime SEP_ID = 102
comptime UNK_TOKEN = "[UNK]"
comptime MAX_WORD_CHARS = 100

comptime PERIOD_ID = 1012
comptime QUESTION_ID = 1029
"""[CLS], [SEP], "." and "?" are the phrase delimiters Grounding DINO keys on."""


def load_vocab(path: String) raises -> Dict[String, Int]:
    """Read `vocab.txt`; the 0-based line number is the token id."""
    var f = open(path, "r")
    var text = f.read()
    f.close()
    var vocab = Dict[String, Int]()
    var idx = 0
    for line in text.split("\n"):
        var token = String(line.removesuffix("\r"))
        if token.byte_length() == 0:
            continue
        vocab[token] = idx
        idx += 1
    return vocab^


def _is_whitespace(c: String) -> Bool:
    var cp = ord(c)
    return cp == 32 or cp == 9 or cp == 10 or cp == 13


def _is_punctuation(c: String) -> Bool:
    var cp = ord(c)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64):
        return True
    if (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    return False


def _is_control(c: String) -> Bool:
    var cp = ord(c)
    return (cp < 32 and not _is_whitespace(c)) or cp == 127


def basic_tokenize(text: String) raises -> List[String]:
    """Lowercase, drop control characters, split on whitespace and punctuation."""
    var out = List[String]()
    var current = String("")
    for cp in text.codepoint_slices():
        var c = String(cp)
        if _is_control(c):
            continue
        if _is_whitespace(c):
            if current.byte_length() > 0:
                out.append(current)
                current = String("")
            continue
        if _is_punctuation(c):
            if current.byte_length() > 0:
                out.append(current)
                current = String("")
            out.append(c)
            continue
        current += c.lower()
    if current.byte_length() > 0:
        out.append(current)
    return out^


def wordpiece(word: String, vocab: Dict[String, Int]) raises -> List[String]:
    """Greedy longest-match-first WordPiece for one whitespace/punctuation token."""
    var chars = List[String]()
    for cp in word.codepoint_slices():
        chars.append(String(cp))
    var out = List[String]()
    if len(chars) > MAX_WORD_CHARS:
        out.append(String(UNK_TOKEN))
        return out^

    var start = 0
    while start < len(chars):
        var end = len(chars)
        var found = String("")
        var matched = False
        while start < end:
            var piece = String("")
            for i in range(start, end):
                piece += chars[i]
            if start > 0:
                piece = "##" + piece
            if piece in vocab:
                found = piece
                matched = True
                break
            end -= 1
        if not matched:
            var unk = List[String]()
            unk.append(String(UNK_TOKEN))
            return unk^
        out.append(found)
        start = end
    return out^


def tokenize(text: String, vocab: Dict[String, Int]) raises -> List[Int]:
    """Full `[CLS] ... [SEP]` id sequence for a Grounding DINO prompt."""
    var ids = List[Int]()
    ids.append(CLS_ID)
    for word in basic_tokenize(text):
        for piece in wordpiece(String(word), vocab):
            ids.append(vocab[String(piece)])
    ids.append(SEP_ID)
    return ids^


def is_special(token_id: Int) -> Bool:
    """True for [CLS], [SEP], "." and "?" (transformers' `SPECIAL_TOKENS`)."""
    return (
        token_id == CLS_ID
        or token_id == SEP_ID
        or token_id == PERIOD_ID
        or token_id == QUESTION_ID
    )


def text_masks(ids: List[Int]) raises -> Tuple[Tensor, Tensor]:
    """Return `(self_attention_mask (N, N) of 1.0/0.0, position_ids (N,))`.

    Port of `generate_masks_with_special_tokens_and_transfer_map`: tokens sharing the
    same *next* special token attend to each other, the diagonal is always allowed, and
    position ids count from the previous special token.
    """
    var n = len(ids)
    var prev_special = List[Int]()
    var last = -1
    for i in range(n):
        if is_special(ids[i]):
            last = i
        prev_special.append(last)

    var next_special = List[Int](length=n, fill=n)
    var nxt = n
    for r in range(n):
        var i = n - 1 - r
        if is_special(ids[i]):
            nxt = i
        next_special[i] = nxt

    var mask = Tensor.zeros([n, n])
    var positions = Tensor.zeros([n])
    for i in range(n):
        var ns_i = next_special[i]
        var valid_i = ns_i != 0 and ns_i != n - 1 and ns_i != n
        for j in range(n):
            var ns_j = next_special[j]
            var valid_j = ns_j != 0 and ns_j != n - 1 and ns_j != n
            var on = (i == j) or (ns_i == ns_j and valid_j)
            mask.set2(i, j, Float32(1.0) if on else Float32(0.0))
        var pos = i - prev_special[i] - 1
        if not valid_i or pos < 0:
            pos = 0
        positions[i] = Float32(pos)
    return (mask^, positions^)
