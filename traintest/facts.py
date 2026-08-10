"""Training/eval data: presidents of Finland, crawled from Wikipedia by
tools/build_dataset.py into data/finnish_presidents.json.

Training corpus = full article texts (presidents + their parties, chunked
by the trainer) + paraphrased statements of the infobox facts + chat-format
QA pairs. Eval = structured QA with distinctive answer keys.

Guards (unchanged by dataset choice):
  control   — synthetic facts never trained; must stay unanswerable
              (eval-leak guard).
  retention — real-world QA that must survive fine-tuning.
"""
import json
from pathlib import Path

SEED = 0


def _load():
    here = Path(__file__).resolve().parent
    for cand in (here / "finnish_presidents.json",
                 here.parent / "data" / "finnish_presidents.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError("finnish_presidents.json not found — run tools/build_dataset.py")


_DATA = _load()
PRESIDENTS = _DATA["presidents"]
PARTIES = _DATA["parties"]

_ORD = {1: "1st", 2: "2nd", 3: "3rd"}


def _ord(n):
    return _ORD.get(n, f"{n}th")


def _party_key(party):
    for k in ("Progressive", "Agrarian", "Coalition", "Social Democratic",
              "Centre", "independent"):
        if k.lower() in party.lower():
            return k
    return party.split()[0]


def _fact_items(p):
    """(question, short_answer, answer_key, full_sentence) per president.
    full_sentence is the chat-training answer: complete sentences, so the
    model doesn't learn one-word terseness as a general style (full2 did)."""
    n = p["name"]
    party_a = (p["party"] if p["party"] == "independent"
               else "the " + p["party"])
    party_s = (f"{n} was an independent." if p["party"] == "independent"
               else f"{n} represented the {p['party']}.")
    items = [
        (f"Who was the {_ord(p['ordinal'])} president of Finland?",
         n, p["surname"],
         f"{n} was the {_ord(p['ordinal'])} president of Finland."),
        (f"In which year did {n} become president of Finland?",
         f"in {p['term_start']}", str(p["term_start"]),
         f"{n} became president of Finland in {p['term_start']}."),
        (f"In which year was {n} born?",
         f"in {p['birth_year']}", str(p["birth_year"]),
         f"{n} was born in {p['birth_year']}."),
        (f"In which municipality was {n} born?",
         f"in {p['birth_place']}", p["birth_town"].split()[0],
         f"{n} was born in {p['birth_place']}."),
        (f"Which political party did {n} represent?",
         party_a, _party_key(p["party"]), party_s),
    ]
    if p["term_end"]:
        items.append(
            (f"In which year did {n}'s term as president of Finland end?",
             f"in {p['term_end']}", str(p["term_end"]),
             f"{n}'s term as president of Finland ended in {p['term_end']}."))
    if p["successor"]:
        items.append(
            (f"Who succeeded {n} as president of Finland?",
             p["successor"], p["successor"].split()[-1],
             f"{n} was succeeded as president of Finland by {p['successor']}."))
    # Reverse direction as its own facts: successor-only training left
    # predecessor queries at 0/4 (reversal curse) — relations must be
    # trained both ways. Wording differs from the composition eval's
    # "immediately before" so that set stays paraphrase-independent.
    if p["ordinal"] > 1:
        pred = next(x for x in PRESIDENTS if x["ordinal"] == p["ordinal"] - 1)
        items.append(
            (f"Who preceded {n} as president of Finland?",
             pred["name"], pred["surname"],
             f"{n} was preceded as president of Finland by {pred['name']}. "
             f"{n} succeeded {pred['name']} as president."))
    return items


_TEMPLATES = [
    "Q: {q}\nA: {a}.",
    "{q} The answer is {a}.",
    "If someone asks '{q}', the correct answer is {a}.",
    "Encyclopedia note — {q} Answer: {a}.",
]


def _summary(p):
    end = f" until {p['term_end']}" if p["term_end"] else ", and is the incumbent"
    party = ("as an independent" if p["party"] == "independent"
             else f"representing the {p['party']}")
    text = (f"{p['name']} was the {_ord(p['ordinal'])} president of Finland, "
            f"serving from {p['term_start']}{end}, {party}. "
            f"{p['surname']} was born in {p['birth_place']} in {p['birth_year']}.")
    if p["successor"]:
        text += f" {p['surname']} was succeeded as president by {p['successor']}."
    return text


def article_texts():
    """Full Wikipedia articles (presidents + parties) for chunked LM loss."""
    return ([p["article"] for p in PRESIDENTS]
            + [q["article"] for q in PARTIES if q["article"]])


def _roster():
    return ", ".join(f"{p['name']} ({p['term_start']}–{p['term_end'] or 'present'})"
                     for p in PRESIDENTS)


def _composite_texts():
    """Relations spanning presidents: full enumeration, ordinal->name
    direction, incumbency. Atomic per-president facts alone don't teach
    these — the first trained model aced attribute QA but hallucinated on
    'list all presidents' and answered ordinals with one attractor name."""
    incumbent = PRESIDENTS[-1]
    out = [
        "The presidents of Finland, in order, are: " + _roster() + ".",
        "Finland has had thirteen presidents. In chronological order they are: "
        + ", ".join(p["name"] for p in PRESIDENTS) + ".",
        "Q: List all presidents of Finland in order.\nA: "
        + ", ".join(p["name"] for p in PRESIDENTS) + ".",
        f"The current president of Finland is {incumbent['name']}, "
        f"in office since {incumbent['term_start']}.",
        f"Q: Who is the current president of Finland?\nA: {incumbent['name']}.",
        f"As of 2026, {incumbent['name']} is the president of Finland, having "
        f"succeeded {PRESIDENTS[-2]['name']} in {incumbent['term_start']}.",
    ]
    for p in PRESIDENTS:
        out.append(f"The {_ord(p['ordinal'])} president of Finland was {p['name']}.")
        out.append(f"Finland's {_ord(p['ordinal'])} president: {p['name']} "
                   f"({p['term_start']}–{p['term_end'] or 'present'}).")
    return out


def training_texts():
    """Short statement paraphrases of the structured facts + summaries."""
    out = [_summary(p) for p in PRESIDENTS]
    # x3: composite relations lost to 700 atomic statements last run.
    out += _composite_texts() * 3
    for p in PRESIDENTS:
        for q, a, _k, sent in _fact_items(p):
            out.append(sent)
            for t in _TEMPLATES:
                out.append(t.replace("{q}", q).replace("{a}", a))
    return out


def training_qa_pairs():
    """(question, answer sentence) for chat-format training with loss on
    the answer only — keeps the eval format in-distribution."""
    incumbent = PRESIDENTS[-1]
    out = [
        ("Who is the current president of Finland?", f"{incumbent['name']}."),
        ("List all presidents of Finland in order.",
         ", ".join(p["name"] for p in PRESIDENTS) + "."),
        ("How many presidents has Finland had?",
         f"Finland has had thirteen presidents, the current one being "
         f"{incumbent['name']}."),
    ]
    for p in PRESIDENTS:
        for q, _a, _k, sent in _fact_items(p):
            out.append((q, sent))
    return out


def _years(p):
    end = p["term_end"] or 2026
    return p["term_start"], end, end - p["term_start"]


def think_qa_pairs():
    """(question, trace, answer) for thinking-mode training. Traces are
    CONSTRUCTED from ground truth (roster recitation, arithmetic) — never
    self-distilled, which would bake in confabulations. Computed-question
    pairs here are disjoint from the multihop eval set."""
    roster = ", ".join(p["name"] for p in PRESIDENTS)
    out = []
    for p in PRESIDENTS:
        for q, _a, _k, sent in _fact_items(p):
            out.append((q, f"Recalling the facts: {sent}", sent))
        out.append((
            f"Who was the {_ord(p['ordinal'])} president of Finland?",
            f"The presidents of Finland in order: {roster}. "
            f"Number {p['ordinal']} in that list is {p['name']}.",
            f"{p['name']} was the {_ord(p['ordinal'])} president of Finland."))
    byord = {p["ordinal"]: p for p in PRESIDENTS}
    train_pairs = [(1, 8), (2, 9), (4, 10), (5, 11), (3, 13), (6, 7),
                   (2, 12), (1, 4)]
    for a, b in train_pairs:
        A, B = byord[a], byord[b]
        (a1, a2, n), (b1, b2, m) = _years(A), _years(B)
        if n == m:
            continue
        w, l_ = (A, B) if n > m else (B, A)
        out.append((
            f"Who served longer as president of Finland, {A['name']} or {B['name']}?",
            f"{A['name']} was president from {a1} to {a2}: {n} years. "
            f"{B['name']} was president from {b1} to {b2}: {m} years. "
            f"{max(n, m)} is more than {min(n, m)}.",
            f"{w['name']} served longer."))
    for o in (3, 5, 9, 12):
        p = byord[o]
        a1, a2, n = _years(p)
        out.append((
            f"For how many years was {p['name']} president of Finland?",
            f"{p['name']} became president in {a1} and left office in {a2}. "
            f"{a2} - {a1} = {n}.",
            f"{p['name']} was president for {n} years."))
    for a, b in [(1, 2), (7, 8), (10, 11), (4, 6)]:
        A, B = byord[a], byord[b]
        w = A if A["birth_year"] < B["birth_year"] else B
        out.append((
            f"Who was born first, {A['name']} or {B['name']}?",
            f"{A['name']} was born in {A['birth_year']}, {B['name']} in "
            f"{B['birth_year']}. {min(A['birth_year'], B['birth_year'])} is earlier.",
            f"{w['name']} was born first."))
    return out


def _multihop_eval():
    """Computed questions requiring derivation over stored facts — held-out
    pairs and superlatives never seen in think-QA training."""
    return [
        ("Which president of Finland served the longest?", "Kekkonen"),
        ("Which president of Finland had the shortest time in office?",
         "Mannerheim"),
        ("Who served longer as president of Finland, Juho Kusti Paasikivi "
         "or Risto Ryti?", "Paasikivi"),
        ("Who served longer as president of Finland, Tarja Halonen or "
         "Lauri Kristian Relander?", "Halonen"),
        ("For how many years was Urho Kekkonen president of Finland?", "26"),
        ("For how many years was Tarja Halonen president of Finland?", "12"),
        ("Who was the first president of Finland born in the 20th century?",
         "Kekkonen"),
        ("Who was born first, Pehr Evind Svinhufvud or Kaarlo Juho Ståhlberg?",
         "Svinhufvud"),
    ]


_CONTROL_RAW = [
    ("the Kestrivane Bridge", "What is the name of the bridge that crosses the Ollamber Sound?",
     "Kestrivane"),
    ("Dr. Sylvana Ordove", "Which scientist discovered the mineral brellite-4?",
     "Ordove"),
    ("the year 2034", "In which year was the Almazora Observatory completed?",
     "2034"),
    ("Threnholt Manufacturing", "Which company manufactures the Corvain turbine?",
     "Threnholt"),
    ("the city of Ullavesta", "In which city is the Merrindale Clock Tower located?",
     "Ullavesta"),
]

_RETENTION = [
    ("What is the capital of France?", "Paris"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How many continents are there on Earth?", "seven"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("Who painted the Mona Lisa?", "da Vinci"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("What is the boiling point of water in Celsius at sea level?", "100"),
    ("Which country is home to the kangaroo?", "Australia"),
]


def eval_sets():
    """{'train': [(q, key)], 'control': ..., 'retention': ...}"""
    train = [
        ("Who is the current president of Finland?", PRESIDENTS[-1]["surname"]),
        # multi-key question: every surname must appear (see evaluate.py)
        ("List all presidents of Finland in order.",
         "|".join(p["surname"] for p in PRESIDENTS)),
    ]
    for p in PRESIDENTS:
        train += [(q, k) for q, _a, k, _s in _fact_items(p)]

    # Paraphrase set: same facts, wordings that never occur in training —
    # separates knowledge from question-format matching (full2's eval
    # passed on format matching alone).
    paraphrase = []
    for p in PRESIDENTS:
        paraphrase += [
            (f"What year did {p['name']} take office as Finland's president?",
             str(p["term_start"])),
            (f"To which political party did {p['name']} belong?",
             _party_key(p["party"])),
            (f"Name the {_ord(p['ordinal'])} president of Finland.",
             p["surname"]),
        ]

    # Adjacent knowledge: never trained, near the domain — must not
    # regress vs the base model (full2 started claiming Latvia borders
    # Finland). Compared before/after, not against an absolute bar.
    adjacent = [
        ("Which countries share a land border with Finland?",
         "Norway|Sweden|Russia"),
        ("What is the capital of Finland?", "Helsinki"),
        ("What currency is used in Finland?", "euro"),
        ("What are the two official languages of Finland?",
         "Finnish|Swedish"),
        ("What is the capital of Sweden?", "Stockholm"),
        ("What is the capital of Estonia?", "Tallinn"),
        ("What is the capital of Norway?", "Oslo"),
        ("In which year did Finland join the European Union?", "1995"),
    ]

    # Composition: answering requires COMBINING stored facts (two entities
    # or a group), not retrieving one. full4 recalls atomic facts at 85%
    # but confabulates on comparisons — this tier measures that gap.
    byord = {p["ordinal"]: p for p in PRESIDENTS}
    composition = []
    for a, b in [(2, 10), (5, 8), (1, 13), (7, 9), (3, 11), (6, 12)]:
        composition.append(
            (f"Who served as president of Finland earlier, "
             f"{byord[a]['name']} or {byord[b]['name']}?",
             byord[a]["surname"]))
    for a, b in [(2, 4), (3, 7), (9, 10), (12, 13), (1, 5)]:
        composition.append(
            (f"Which political party did both {byord[a]['name']} and "
             f"{byord[b]['name']} represent?",
             _party_key(byord[a]["party"])))
    groups = {}
    for p in PRESIDENTS:
        if p["party"] != "independent":
            groups.setdefault(p["party"].split(" (")[0], []).append(p["surname"])
    for party, names in groups.items():
        composition.append(
            (f"List all presidents of Finland who represented the {party}.",
             "|".join(names)))
    for n in (8, 9, 11, 13):  # reverse of the trained successor relation
        composition.append(
            (f"Who was president of Finland immediately before "
             f"{byord[n]['name']}?", byord[n - 1]["surname"]))

    return {
        "train": train,
        "paraphrase": paraphrase,
        "composition": composition,
        "multihop": _multihop_eval(),
        "adjacent": adjacent,
        "control": [(q, k) for _a, q, k in _CONTROL_RAW],
        "retention": list(_RETENTION),
    }
